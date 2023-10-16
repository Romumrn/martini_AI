# -*- coding: utf-8 -*-

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
import torch
from martini_AI import create_folder_and_download_files , get_T5_model, get_prediction


torch.cuda.empty_cache()

#@title Import dependencies and check whether GPU is available. { display-mode: "form" }

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

create_folder_and_download_files()
#@title Load the checkpoint for secondary structure prediction. { display-mode: "form" }
preloaded_model = get_T5_model()

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

def is_fasta(file_path):
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline()
            return first_line.startswith(">")
    except FileNotFoundError:
        return False

# Load example fasta.
if sys.argv[1] :
    if is_fasta(sys.argv[1]):
        seqs  =read_fasta(sys.argv[1])
        id = seqs.keys()[0]
        seq = seqs[id]
    else:
       seq = sys.argv[1]
        if sys.argv[2] :
            id = sys.argv[2]
        else:
            id = "seq000"
else:
    print( "Please provide a sequence")
    exit


# Compute embeddings and/or secondary structure predictions
print( "get prediction")

result = get_prediction(preloaded_model, seq, id)



# Mapping of DSSP8 values to secondary structure classes
# 310 helix (G), α-helix (H), π-helix (I), β-strand (E),
# bridge (B), turn (T), bend (S), and others (C).
dssp8 = ["G", "H", "I", "E", "B", "T", "S", "C"]

# convert array into dataframe
# DF = pd.DataFrame(result)
# DF.columns = ["AA"] + dssp8
# DF.pop(DF.columns[0])
# max_index = DF.astype(float).idxmax(axis=1)
# list_predicted = max_index.tolist()
DF = pd.DataFrame(result)
DF.columns = ["AA"] + dssp8

filename=seq+"_prediction.csv"
DF.to_csv(filename, index=False)  

print( DF.describe() )

print( f"Save in {filename}")
