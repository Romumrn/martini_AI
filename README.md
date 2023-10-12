# Project: Martini Dyhedrals powered by AI


The aim of this repo is to be able to predict the probability of secondary structure conformation based on the sequence of amino acid only with a Artifical intelligence model: [ProtTrans](https://github.com/agemagician/ProtTrans). Once these probabilities computed, give personalised dyhedral angle potentials for each amino acid based on sequence information. This will make it possible to make proteins more flexible/regionally oriented and thus ensure better folding. 

## Requirements

- Librairies : please find all information in [requirements](./requirements.txt)
- If local use : a GPU with more than 12Gb of memory


# How to run 

## Command line
```
pip install -r requirements.txt

python run_ProtT5.py "SEQUENCE" ID

```

## Or run it google collab 


check the notebook link (soon...)