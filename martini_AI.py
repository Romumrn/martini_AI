from transformers import T5EncoderModel, T5Tokenizer
import torch
import h5py
import numpy as np
import pandas as pd
import time
import os
import wget
from urllib.request import Request, urlopen

# Clear GPU cache if CUDA is available
torch.cuda.empty_cache()

# Check if GPU is available, if not, use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device used for the model: {}".format(device))

# Function to create folders and download necessary files if they don't exist
def create_folder_and_download_files(path='.'):
    """
    This function prepares the execution of the ML model.
    """
    if not os.path.exists(os.path.join(path, "protT5")):
        os.mkdir(os.path.join(path, "protT5"))

    if not os.path.exists(os.path.join(path, "protT5/protT5_checkpoint")):
        os.mkdir(os.path.join(path, "protT5/protT5_checkpoint"))

    if not os.path.exists(os.path.join(path, "protT5/sec_struct_checkpoint")):
        os.mkdir(os.path.join(path, "protT5/sec_struct_checkpoint"))

    if not os.path.exists(os.path.join(path, "protT5/output")):
        os.mkdir(os.path.join(path, "protT5/output"))

    # Download files if they don't exist
    if not os.path.exists("protT5/sec_struct_checkpoint/secstruct_checkpoint.pt"):
        print("Downloading secstruct_checkpoint.pt")
        # Set a User-Agent header for the HTTP request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

        # Create an HTTP request with the User-Agent header
        request = Request(
            "http://data.bioembeddings.com/public/embeddings/feature_models/t5/secstruct_checkpoint.pt", headers=headers)

        # Download the file and save it
        with urlopen(request) as response, open("protT5/sec_struct_checkpoint/secstruct_checkpoint.pt", 'wb') as out_file:
            out_file.write(response.read())

# Define a convolutional neural network class
class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # CNN layers for feature extraction
        self.elmo_feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
        )
        n_final_in = 32
        # Convolutional layers for secondary structure prediction
        self.dssp3_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 3, kernel_size=(7, 1), padding=(3, 0))  # 7
        )
        self.dssp8_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 8, kernel_size=(7, 1), padding=(3, 0))
        )
        self.diso_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 2, kernel_size=(7, 1), padding=(3, 0))
        )

    def forward(self, x):
        # Input reshaping
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)
        # Extract features using CNN
        x = self.elmo_feature_extractor(x)
        # Predict secondary structure
        d3_Yhat = self.dssp3_classifier(x).squeeze(dim=-1).permute(0, 2, 1)
        d8_Yhat = self.dssp8_classifier(x).squeeze(dim=-1).permute(0, 2, 1)
        diso_Yhat = self.diso_classifier(x).squeeze(dim=-1).permute(0, 2, 1)
        return d3_Yhat, d8_Yhat, diso_Yhat

# Function to load the checkpoint for secondary structure prediction
def load_sec_struct_model(checkpoint_dir):
    state = torch.load(checkpoint_dir)
    model = ConvNet()
    model.load_state_dict(state['state_dict'])
    model = model.eval()
    model = model.to(device)
    #print('Loaded sec. struct. model from epoch: {:.1f}'.format(state['epoch']))
    return model

# Function to load the encoder-part of ProtT5 in half-precision
def get_T5_model():
    model = T5EncoderModel.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device)  # Move model to GPU
    model = model.eval()  # Set model to evaluation mode
    tokenizer = T5Tokenizer.from_pretrained(
        'Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    return (model, tokenizer)

# Function to read a fasta file and extract sequences
def read_fasta(fasta_path, split_char="!", id_field=0):
    '''
    Reads in fasta file containing multiple sequences.
    Split_char and id_field allow to control identifier extraction from header.
    E.g.: set split_char="|" and id_field=1 for SwissProt/UniProt Headers.
    Returns a dictionary holding multiple sequences or only a single sequence, depending on the input file.
    '''
    seqs = dict()
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            # Get UniProt ID from header and create a new entry
            if line.startswith('>'):
                uniprot_id = line.replace(
                    '>', '').strip().split(split_char)[id_field]
                # Replace tokens that are misinterpreted when loading h5
                uniprot_id = uniprot_id.replace("/", "_").replace(".", "_")
                seqs[uniprot_id] = ''
            else:
                # Replace all white-space characters, join sequences spanning multiple lines, drop gaps, and cast to uppercase
                seq = ''.join(line.split()).upper().replace("-", "")
                # Replace all non-standard AAs and map them to unknown/X
                seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
                seqs[uniprot_id] += seq
    example_id = next(iter(seqs))
    print("Read {} sequences.".format(len(seqs)))
    return seqs


def get_prediction(loaded_model, sequence, pdb_id='seq_0',temperature=1, max_residues=4000, max_seq_len=1000):
    """
    This function performs secondary structure prediction for a given protein sequence.

    Parameters:
    - loaded_model: Tuple containing the ProtT5 model and tokenizer.
    - sequence: Protein sequence as a string.
    - pdb_id: Identifier for the protein sequence (default: 'seq_0').
    - temperature: Control the "smoothness" of the softmax distribution. (default: 1)
    - max_residues: Maximum number of residues in a batch (default: 4000).
    - max_seq_len: Maximum sequence length (default: 1000).

    Returns:
    - result: Numpy array containing the sequence and predicted secondary structure probabilities.
              Each row consists of a residue and its associated probabilities for different secondary structure classes.
    """
    if len(sequence) > max_seq_len:
        print( f"Error: The protein is to long (max_residue:{max_seq_len}).")
        return 
    model, tokenizer = loaded_model

    # Load the pre-trained secondary structure prediction model
    sec_struct_model = load_sec_struct_model("./protT5/sec_struct_checkpoint/secstruct_checkpoint.pt")

    # Initialize results dictionary to store predictions
    results = {"residue_embs": [], "sec_structs": [], "ss8_tensor": []}

    # Initialize batch list with the input sequence
    start = time.time()
    batch = list()
    seq_len = len(sequence)
    seq = ' '.join(list(sequence))
    batch.append((pdb_id, seq, seq_len))

    # Add special tokens to the input sequence and encode it
    token_encoding = tokenizer.batch_encode_plus(
        [seq], add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(token_encoding['input_ids']).to(device)
    attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

    try:
        with torch.no_grad():
            # Perform embedding of the input sequence using the ProtT5 model
            embedding_repr = model(input_ids, attention_mask=attention_mask)
    except RuntimeError:
        print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))

    # Predict secondary structure using the loaded sec_struct_model
    d3_Yhat, d8_Yhat, diso_Yhat = sec_struct_model(embedding_repr.last_hidden_state)

    # Slice off the padding in the predictions
    emb = embedding_repr.last_hidden_state[0, :seq_len]

    # Store predicted secondary structure labels, residue embeddings, and softmax probabilities
    results["sec_structs"] = torch.max(d3_Yhat[0, :seq_len], dim=1)[1].detach().cpu().numpy().squeeze()
    results["residue_embs"] = emb.detach().cpu().numpy().squeeze()
    
    # Apply softmax with temperature to d8_Yhat and store the result
    probs = torch.nn.Softmax(dim=1)
    results["ss8_tensor"] = probs(d8_Yhat[0, :seq_len]/temperature)
    
    # Calculate the time taken for prediction
    passed_time = time.time() - start
    print('### Done in {} sec'.format(str(passed_time)))

    # Convert 'probabilities' to a float numpy array
    probabilities = np.array(results["ss8_tensor"].cpu().detach().numpy(), dtype=np.float32)

    # Stack 'sequence' and 'probabilities' as columns in the result array
    result = np.column_stack((list(sequence), probabilities))

    return result



#####  Need to fix it #####
def get_predictions(loaded_model, sequences, pdb_ids=None, temperature=1, max_residues=4000, max_seq_len=1000, max_batch=100):
    """
    This function performs secondary structure prediction for a batch of protein sequences.

    Parameters:
    - loaded_model: Tuple containing the ProtT5 model and tokenizer.
    - sequences: List of protein sequences as strings.
    - pdb_ids: List of identifiers for the protein sequences (default: None).
    - temperature: Control the "smoothness" of the softmax distribution (default: 1).
    - max_residues: Maximum number of residues in a batch (default: 4000).
    - max_seq_len: Maximum sequence length (default: 1000).
    - max_batch: Maximum batch size (default: 100).

    Returns:
    - results: A dictionary containing the predictions and information for each protein sequence.
               Keys include 'residue_embs', 'sec_structs', 'df_ss8_probs', and 'ss8_tensor'.
    """
    model, tokenizer = loaded_model

    # Load the pre-trained secondary structure prediction model
    sec_struct_model = load_sec_struct_model("./protT5/sec_struct_checkpoint/secstruct_checkpoint.pt")

    # Initialize results dictionary to store predictions
    results = {"residue_embs": dict(), "sec_structs": dict(), "df_ss8_probs": dict(), "ss8_tensor": dict()}

    # Create a dictionary to map protein identifiers to their sequences
    seqs = dict()
    if isinstance(sequences, str):
        sequences = [sequences]
    if len(sequences) != len(pdb_ids):
        if pdb_ids is not None:
            print("List of sequences and list of identifiers are not the same length")
            return
    for i in range(len(sequences)):
        if pdb_ids is not None:
            seqs[pdb_ids[i]] = sequences[i]
        else:
            seqs[i] = sequences[i]

    # Sort sequences according to length (reduces unnecessary padding, speeds up embedding)
    seq_dict = sorted(seqs.items(), key=lambda kv: len(seqs[kv[0]]), reverse=True)

    # Initialize a list to hold sequences in the current batch
    batch = list()
    start = time.time()

    for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))

        # Count residues in the current batch and add the last sequence length to
        # avoid processing batches with too many residues (n_res_batch > max_residues)
        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len

        # Check if batch size or sequence length limits are reached, or it's the last sequence
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # Add special tokens, encode sequences, and move to GPU (if available)
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    # Perform embedding of the input sequence using the ProtT5 model
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            d3_Yhat, d8_Yhat, diso_Yhat = sec_struct_model(embedding_repr.last_hidden_state)

            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                # Slice off padding in the embeddings
                emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
                results["sec_structs"][identifier] = torch.max(d3_Yhat[batch_idx, :s_len], dim=1)[1].detach().cpu().numpy().squeeze()
                results["residue_embs"][identifier] = emb.detach().cpu().numpy().squeeze()

                # Apply softmax with temperature to d8_Yhat and store the result
                probs = torch.nn.Softmax(dim=1)
                ss8_tensor_softmaxed = probs(d8_Yhat[batch_idx, :seq_len] / temperature)
                results["ss8_tensor"][identifier] = ss8_tensor_softmaxed
                sequence = seq.split(" ")
                np_probs = np.array(ss8_tensor_softmaxed.cpu().detach().numpy(), dtype=np.float32)
                results["df_ss8_probs"][identifier] = np.column_stack((list(sequence), np_probs))

    # Calculate the time taken for prediction
    #passed_time = time.time() - start
    #print('### Done in {} sec'.format(str(passed_time)))

    return results["df_ss8_probs"]
