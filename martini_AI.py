f# Import necessary libraries and modules
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
    print('Loaded sec. struct. model from epoch: {:.1f}'.format(state['epoch']))
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

def get_prediction(loaded_model, sequence, pdb_id='seq_0', max_residues=4000, max_seq_len=1000, max_batch=100):
    """
    This function performs secondary structure prediction for a given protein sequence.

    Parameters:
    - loaded_model: Tuple containing the ProtT5 model and tokenizer.
    - sequence: Protein sequence as a string.
    - pdb_id: Identifier for the protein sequence (default: 'seq_0').
    - max_residues: Maximum number of residues in a batch (default: 4000).
    - max_seq_len: Maximum sequence length (default: 1000).
    - max_batch: Maximum batch size (default: 100).

    Returns:
    - result: Numpy array containing the sequence and predicted secondary structure probabilities.
              Each row consists of a residue and its associated probabilities for different secondary structure classes.
    """
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
    probs = torch.nn.Softmax(dim=1)
    results["ss8_tensor"] = probs(d8_Yhat[0, :seq_len])

    # Calculate the time taken for prediction
    passed_time = time.time() - start
    print('### Done in {} sec'.format(str(passed_time)))

    # Convert 'probabilities' to a float numpy array
    probabilities = np.array(results["ss8_tensor"].cpu().detach().numpy(), dtype=np.float32)

    # Stack 'sequence' and 'probabilities' as columns in the result array
    result = np.column_stack((list(sequence), probabilities))

    return result


