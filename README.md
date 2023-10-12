# Project: Martini Dyhedrals powered by AI


The aim of this repo is to be able to predict the probability of secondary structure conformation based on the sequence of amino acid only with a Artifical intelligence model: [ProtTrans](https://github.com/agemagician/ProtTrans). Once these probabilities computed, give personalised dyhedral angle potentials for each amino acid based on sequence information. This will make it possible to make proteins more flexible/regionally oriented and thus ensure better folding. 

## Requirements


Before using this code, ensure that you have the following prerequisites in place:

- Python 3.7 or later
-  PyTorch
-   Transformers library
-   H5py
-   Numpy
-   Pandas

Please find all information in [requirements](./requirements.txt)

If you plan to run this code locally, you need access to a GPU with at least 5.2GB of memory.

# How to Run Localy

## Command Line
1. Clone the repository to your local machine.

   ```bash
   git clone https://github.com/Romumrn/martini_AI
   cd martini_AI
   ```
2. Install the required Python libraries using pip.

```
pip install -r requirements.txt
```

3. Run the code for protein secondary structure prediction. You can provide a sequence as a command-line argument.

```
python run_ProtT5.py "SEQUENCE" "ID"
```


## Run in Google Colab

You can also try the test case provided on [google collab](https://colab.research.google.com/drive/1Vo4T-fBKBtFwa6Vj04NCtF9t8uQkA05V)

## Run in Jupyter notebook 
(Not tested yes)

