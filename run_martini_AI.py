# -*- coding: utf-8 -*-
import argparse
import torch
import pandas as pd
from martini_AI import create_folder_and_download_files, get_T5_model, get_prediction
from urllib.request import Request, urlopen
from transformers import T5EncoderModel
from tqdm import tqdm
import wget
import h5py

# Check whether GPU is available.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Function to read a FASTA file and extract sequences.
def read_fasta(fasta_path, split_char="!", id_field=0):
    '''
    Reads in a FASTA file containing multiple sequences. 
    Allows control over identifier extraction from the header.
    Returns a dictionary holding sequences.
    '''

    sequences = dict()
    with open(fasta_path, 'r') as fasta_file:
        for line in fasta_file:
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                uniprot_id = uniprot_id.replace("/", "_").replace(".", "_")
                sequences[uniprot_id] = ''
            else:
                seq = ''.join(line.split()).upper().replace("-", "")
                seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
                sequences[uniprot_id] += seq
    return sequences

# Function to check if a file is in FASTA format.
def is_fasta(file_path):
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline()
            return first_line.startswith(">")
    except FileNotFoundError:
        return False

# Parse command-line arguments.
parser = argparse.ArgumentParser(description="Martini Dihedrals AI-Powered Sequence Analysis")
parser.add_argument("--sequence", "-s", help="Input amino acid sequence for analysis.")
parser.add_argument("--sequence_id", "-id", help="Sequence identifier (optional). Default: 'seq000'", default="seq000")
parser.add_argument("--fasta", "-f", help="FASTA file containing an amino acid sequence (only one sequence is processed at the moment).")
args = parser.parse_args()

# Check the arguments and take appropriate actions.
if args.fasta:
    if not is_fasta(args.fasta):
        print("Error: The provided file does not appear to be a valid FASTA file.")
        quit()
    else:
        seqs = read_fasta(args.fasta)
        id = list(seqs.keys())[0]
        seq = seqs[id]
        print(f"Fasta sequence loaded with identifier: {id}")
elif args.sequence:
    seq = args.sequence
    id = args.sequence_id
else:
    print("Please provide input data (amino acid sequence or FASTA file) for analysis.")
    quit()

# Create a folder and download required files.
create_folder_and_download_files()

# Load the checkpoint for secondary structure prediction.
preloaded_model = get_T5_model()

# Compute embeddings and/or secondary structure predictions.
print("Getting prediction...")
result = get_prediction(preloaded_model, seq, id)

# Mapping of DSSP8 values to secondary structure classes.
dssp8 = ["G", "H", "I", "E", "B", "T", "S", "C"]

# Convert the array into a DataFrame and save the results to a CSV file.
DF = pd.DataFrame(result)
DF.columns = ["AA"] + dssp8

filename = f"{id}_prediction.csv"
DF.to_csv(filename, index=False)

print("Statistics for secondary structure prediction:")
print(DF.describe())
print(f"Results saved in {filename}")
