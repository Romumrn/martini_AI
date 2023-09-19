from transformers import T5EncoderModel, T5Tokenizer
import torch
import h5py
import numpy as np
import pandas as pd
import time
import sys
import os
import wget
from urllib.request import Request, urlopen
torch.cuda.empty_cache()

#@title Import dependencies and check whether GPU is available. { display-mode: "form" }

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using {}".format(device))

# Convolutional neural network (two convolutional layers) to predict secondary structure
class ConvNet( torch.nn.Module ):
    def __init__( self ):
        super(ConvNet, self).__init__()
        # This is only called "elmo_feature_extractor" for historic reason
        # CNN weights are trained on ProtT5 embeddings
        self.elmo_feature_extractor = torch.nn.Sequential(
                        torch.nn.Conv2d( 1024, 32, kernel_size=(7,1), padding=(3,0) ), # 7x32
                        torch.nn.ReLU(),
                        torch.nn.Dropout( 0.25 ),
                        )
        n_final_in = 32
        self.dssp3_classifier = torch.nn.Sequential(
                        torch.nn.Conv2d( n_final_in, 3, kernel_size=(7,1), padding=(3,0)) # 7
                        )

        self.dssp8_classifier = torch.nn.Sequential(
                        torch.nn.Conv2d( n_final_in, 8, kernel_size=(7,1), padding=(3,0))
                        )
        self.diso_classifier = torch.nn.Sequential(
                        torch.nn.Conv2d( n_final_in, 2, kernel_size=(7,1), padding=(3,0))
                        )


    def forward( self, x):
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0,2,1).unsqueeze(dim=-1)
        x         = self.elmo_feature_extractor(x) # OUT: (B x 32 x L x 1)
        d3_Yhat   = self.dssp3_classifier( x ).squeeze(dim=-1).permute(0,2,1) # OUT: (B x L x 3)
        d8_Yhat   = self.dssp8_classifier( x ).squeeze(dim=-1).permute(0,2,1) # OUT: (B x L x 8)
        diso_Yhat = self.diso_classifier(  x ).squeeze(dim=-1).permute(0,2,1) # OUT: (B x L x 2)

        # print( "print( d8_Yhat ) " )
        # for i in d8_Yhat:
        #   print( i)
        return d3_Yhat, d8_Yhat, diso_Yhat

#@title Load the checkpoint for secondary structure prediction. { display-mode: "form" }
def load_sec_struct_model():
  checkpoint_dir="./protT5/sec_struct_checkpoint/secstruct_checkpoint.pt"
  state = torch.load( checkpoint_dir )
  model = ConvNet()
  model.load_state_dict(state['state_dict'])
  model = model.eval()
  model = model.to(device)
  print('Loaded sec. struct. model from epoch: {:.1f}'.format(state['epoch']))

  return model

#@title Load encoder-part of ProtT5 in half-precision. { display-mode: "form" }
# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50)
def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer

#@title Read in file in fasta format. { display-mode: "form" }
def read_fasta( fasta_path, split_char="!", id_field=0):
    '''
        Reads in fasta file containing multiple sequences.
        Split_char and id_field allow to control identifier extraction from header.
        E.g.: set split_char="|" and id_field=1 for SwissProt/UniProt Headers.
        Returns dictionary holding multiple sequences or only single
        sequence, depending on input file.
    '''

    seqs = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                seqs[ uniprot_id ] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines, drop gaps and cast to upper-case
                seq= ''.join( line.split() ).upper().replace("-","")
                # repl. all non-standard AAs and map them to unknown/X
                seq = seq.replace('U','X').replace('Z','X').replace('O','X')
                seqs[ uniprot_id ] += seq
    example_id=next(iter(seqs))
    print("Read {} sequences.".format(len(seqs)))
    print("Example:\n{}\n{}".format(example_id,seqs[example_id]))

    return seqs


#@title Generate embeddings. { display-mode: "form" }
# Generate embeddings via batch-processing
# per_residue indicates that embeddings for each residue in a protein should be returned.
# per_protein indicates that embeddings for a whole protein should be returned (average-pooling)
# max_residues gives the upper limit of residues within one batch
# max_seq_len gives the upper sequences length for applying batch-processing
# max_batch gives the upper number of sequences per batch
def get_embeddings( model, tokenizer, seqs, per_residue, per_protein, sec_struct, get_SS8, max_residues=4000, max_seq_len=1000, max_batch=100 ):

    if sec_struct:
      sec_struct_model = load_sec_struct_model()

    results = {"residue_embs" : dict(),
               "protein_embs" : dict(),
               "sec_structs" : dict() ,
               "ss8_tensor" : dict()
               }

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict   = sorted( seqs.items(), key=lambda kv: len( seqs[kv[0]] ), reverse=True )
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict,1):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id,seq,seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed
        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            if sec_struct: # in case you want to predict secondary structure from embeddings
              d3_Yhat, d8_Yhat, diso_Yhat = sec_struct_model(embedding_repr.last_hidden_state)


            for batch_idx, identifier in enumerate(pdb_ids): # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                # slice off padding --> batch-size x seq_len x embedding_dim
                emb = embedding_repr.last_hidden_state[batch_idx,:s_len]
                if sec_struct: # get classification results
                    results["sec_structs"][identifier] = torch.max( d3_Yhat[batch_idx,:s_len], dim=1 )[1].detach().cpu().numpy().squeeze()
                if per_residue: # store per-residue embeddings (Lx1024)
                    results["residue_embs"][ identifier ] = emb.detach().cpu().numpy().squeeze()
                if per_protein: # apply average-pooling to derive per-protein embeddings (1024-d)
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()
                if get_SS8:
                    probs = torch.nn.Softmax(dim=1)
                    results["ss8_tensor"][identifier] = probs(d8_Yhat[batch_idx,:s_len])


    passed_time=time.time()-start
    avg_time = passed_time/len(results["residue_embs"]) if per_residue else passed_time/len(results["protein_embs"])
    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
        passed_time/60, avg_time ))
    print('\n############# END #############')
    return results

#@title Write embeddings to disk. { display-mode: "form" }
def save_embeddings(emb_dict,out_path):
    with h5py.File(str(out_path), "w") as hf:
        for sequence_id, embedding in emb_dict.items():
            # noinspection PyUnboundLocalVariable
            hf.create_dataset(sequence_id, data=embedding)
    return None


#@title Write predictions to disk. { display-mode: "form" }
def write_prediction_fasta(predictions, out_path):
  class_mapping = {0:"H",1:"E",2:"L"}
  with open(out_path, 'w+') as out_f:
      out_f.write( '\n'.join(
          [ ">{}\n{}".format(
              seq_id, ''.join( [class_mapping[j] for j in yhat] ))
          for seq_id, yhat in predictions.items()
          ]
            ) )
  return None


def create_folder_and_download(path = '.'):
    # Check if directories exist before creating them
    if not os.path.exists(os.path.join(path ,"protT5")):
        os.mkdir(os.path.join(path ,"protT5"))

    if not os.path.exists(os.path.join(path ,"protT5/protT5_checkpoint")):
        os.mkdir(os.path.join(path ,"protT5/protT5_checkpoint"))

    if not os.path.exists(os.path.join(path ,"protT5/sec_struct_checkpoint")):
        os.mkdir(os.path.join(path ,"protT5/sec_struct_checkpoint"))

    if not os.path.exists(os.path.join(path ,"protT5/output")):
        os.mkdir(os.path.join(path ,"protT5/output"))
        
    
    # Download files
    if not os.path.exists("protT5/sec_struct_checkpoint/secstruct_checkpoint.pt"):
        print( "downloading secstruct_checkpoint.pt")
        # Set a User-Agent header
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

        # Create a request with the User-Agent header
        request = Request("http://data.bioembeddings.com/public/embeddings/feature_models/t5/secstruct_checkpoint.pt", headers=headers)

        # Download the file
        with urlopen(request) as response, open("protT5/sec_struct_checkpoint/secstruct_checkpoint.pt", 'wb') as out_file:
            out_file.write(response.read())
        #wget.download("http://data.bioembeddings.com/public/embeddings/feature_models/t5/secstruct_checkpoint.pt", "protT5/sec_struct_checkpoint/secstruct_checkpoint.pt")


# MAKE A CONDITION TO CHECK
create_folder_and_download()

# whether to retrieve embeddings for each residue in a protein
# --> Lx1024 matrix per protein with L being the protein's length
# as a rule of thumb: 1k proteins require around 1GB RAM/disk
per_residue = False
per_residue_path = "./protT5/output/per_residue_embeddings.h5" # where to store the embeddings

# whether to retrieve per-protein embeddings
# --> only one 1024-d vector per protein, irrespective of its length
per_protein = True
per_protein_path = "./protT5/output/per_protein_embeddings.h5" # where to store the embeddings

# whether to retrieve secondary structure predictions
# This can be replaced by your method after being trained on ProtT5 embeddings
sec_struct = True
sec_struct_path = "./protT5/output/ss3_preds.fasta" # file for storing predictions

# make sure that either per-residue or per-protein embeddings are stored
assert per_protein is True or per_residue is True or sec_struct is True, print(
    "Minimally, you need to active per_residue, per_protein or sec_struct. (or any combination)")

# Load the encoder part of ProtT5-XL-U50 in half-precision (recommended)
model, tokenizer = get_T5_model()

# Load example fasta.
if sys.argv[1] :
    seqs = read_fasta( sys.argv[1] )
    print( "fasta loaded")
else:
    print( "Please provide a fasta file")
    exit

# Compute embeddings and/or secondary structure predictions
print( "get prediction")
results = get_embeddings( model, tokenizer, seqs, per_residue, per_protein, sec_struct, True)

# List of protein names extracted from the 'seqs' dictionary
list_prot_name = list(seqs.keys())

# Mapping of DSSP8 values to secondary structure classes
# 310 helix (G), α-helix (H), π-helix (I), β-strand (E),
# bridge (B), turn (T), bend (S), and others (C).
dssp8 = ["G", "H", "I", "E", "B", "T", "S", "C"]

results_array = {}
# Loop through each protein name
for i in list_prot_name:
    print(f"Process {i}")
    sequence = seqs[i]
    probabilities = results["ss8_tensor"][i].cpu().detach().numpy()  # Convert tensor to NumPy array
    result = np.column_stack((list(sequence), probabilities))  # Combine sequence and probabilities
    results_array[i] = result
    
    # convert array into dataframe
    DF = pd.DataFrame(results_array[i])
    name = i.replace(" ","_")
    # save the dataframe as a csv file
    DF.to_csv(f"data_{name}.csv")

print( results_array )