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



def create_folder_and_download_files(path='.'):
    """
    This function will prepare the execution of the ML model.
    """
    if not os.path.exists(os.path.join(path, "protT5")):
        os.mkdir(os.path.join(path, "protT5"))

    if not os.path.exists(os.path.join(path, "protT5/protT5_checkpoint")):
        os.mkdir(os.path.join(path, "protT5/protT5_checkpoint"))

    if not os.path.exists(os.path.join(path, "protT5/sec_struct_checkpoint")):
        os.mkdir(os.path.join(path, "protT5/sec_struct_checkpoint"))

    if not os.path.exists(os.path.join(path, "protT5/output")):
        os.mkdir(os.path.join(path, "protT5/output"))

    # Download files
    if not os.path.exists("protT5/sec_struct_checkpoint/secstruct_checkpoint.pt"):
        print("downloading secstruct_checkpoint.pt")
        # Set a User-Agent header
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

        # Create a request with the User-Agent header
        request = Request(
            "http://data.bioembeddings.com/public/embeddings/feature_models/t5/secstruct_checkpoint.pt", headers=headers)

        # Download the file
        with urlopen(request) as response, open("protT5/sec_struct_checkpoint/secstruct_checkpoint.pt", 'wb') as out_file:
            out_file.write(response.read())
        #wget.download("http://data.bioembeddings.com/public/embeddings/feature_models/t5/secstruct_checkpoint.pt", "protT5/sec_struct_checkpoint/secstruct_checkpoint.pt")


# 
# print("Using {}".format(device))


class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # This is only called "elmo_feature_extractor" for historic reason
        # CNN weights are trained on ProtT5 embeddings
        self.elmo_feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 32, kernel_size=(
                7, 1), padding=(3, 0)),  # 7x32
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
        )
        n_final_in = 32
        self.dssp3_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 3, kernel_size=(
                7, 1), padding=(3, 0))  # 7
        )

        self.dssp8_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 8, kernel_size=(7, 1), padding=(3, 0))
        )
        self.diso_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 2, kernel_size=(7, 1), padding=(3, 0))
        )

    def forward(self, x):
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)
        x = self.elmo_feature_extractor(x)  # OUT: (B x 32 x L x 1)
        d3_Yhat = self.dssp3_classifier(x).squeeze(
            dim=-1).permute(0, 2, 1)  # OUT: (B x L x 3)
        d8_Yhat = self.dssp8_classifier(x).squeeze(
            dim=-1).permute(0, 2, 1)  # OUT: (B x L x 8)
        diso_Yhat = self.diso_classifier(x).squeeze(
            dim=-1).permute(0, 2, 1)  # OUT: (B x L x 2)

        # print( "print( d8_Yhat ) " )
        # for i in d8_Yhat:
        #   print( i)
        return d3_Yhat, d8_Yhat, diso_Yhat

# @title Load the checkpoint for secondary structure prediction. { display-mode: "form" }


def load_sec_struct_model(checkpoint_dir):
    state = torch.load(checkpoint_dir)
    model = ConvNet()
    model.load_state_dict(state['state_dict'])
    model = model.eval()
    model = model.to(device)
    print('Loaded sec. struct. model from epoch: {:.1f}'.format(
        state['epoch']))
    return model

# @title Load encoder-part of ProtT5 in half-precision. { display-mode: "form" }
# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50)


def get_T5_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device used for the model : {}".format(device))
    model = T5EncoderModel.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device)  # move model to GPU
    model = model.eval()  # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained(
        'Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    return (model, tokenizer)


def read_fasta(fasta_path, split_char="!", id_field=0):
    '''
        Reads in fasta file containing multiple sequences.
        Split_char and id_field allow to control identifier extraction from header.
        E.g.: set split_char="|" and id_field=1 for SwissProt/UniProt Headers.
        Returns dictionary holding multiple sequences or only single
        sequence, depending on input file.
    '''

    seqs = dict()
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace(
                    '>', '').strip().split(split_char)[id_field]
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/", "_").replace(".", "_")
                seqs[uniprot_id] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines, drop gaps and cast to upper-case
                seq = ''.join(line.split()).upper().replace("-", "")
                # repl. all non-standard AAs and map them to unknown/X
                seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
                seqs[uniprot_id] += seq
    example_id = next(iter(seqs))
    print("Read {} sequences.".format(len(seqs)))
    print("Example:\n{}\n{}".format(example_id, seqs[example_id]))

    return seqs


def get_prediction(loaded_model, sequence, pdb_id='seq_0', max_residues=4000, max_seq_len=1000, max_batch=100):
    """
    Blabla
    """
    a = time.time()
    model, tokenizer = loaded_model
    b = time.time()
    print(b - a)
    sec_struct_model = load_sec_struct_model()
    print(time.time()-b)
    results = {"residue_embs": [], "protein_embs": [], "sec_structs": [], "ss8_tensor": []
               }
    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    start = time.time()
    batch = list()
    seq_len = len(sequence)
    seq = ' '.join(list(sequence))
    batch.append((pdb_id, seq, seq_len))

    # count residues in current batch and add the last sequence length to
    # avoid that batches with (n_res_batch > max_residues) get processed
    n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len

    # add_special_tokens adds extra token at the end of each sequence
    token_encoding = tokenizer.batch_encode_plus(
        [seq], add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(token_encoding['input_ids']).to(device)
    attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)
    print(3)
    try:
        with torch.no_grad():
            # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
            embedding_repr = model(input_ids, attention_mask=attention_mask)
    except RuntimeError:
        print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))

    d3_Yhat, d8_Yhat, diso_Yhat = sec_struct_model(
        embedding_repr.last_hidden_state)
    print(4)
    # slice off padding --> batch-size x seq_len x embedding_dim
    emb = embedding_repr.last_hidden_state[0, :seq_len]
    results["sec_structs"] = torch.max(d3_Yhat[0, :seq_len], dim=1)[
        1].detach().cpu().numpy().squeeze()
    results["residue_embs"] = emb.detach().cpu().numpy().squeeze()
    probs = torch.nn.Softmax(dim=1)
    results["ss8_tensor"] = probs(d8_Yhat[0, :seq_len])

    passed_time = time.time()-start
    print('### Done in {} sec'.format(str(passed_time)))
    # Convert tensor to NumPy array
    probabilities = results["ss8_tensor"].cpu().detach().numpy()
    result = np.column_stack((list(sequence), probabilities))
    return result

