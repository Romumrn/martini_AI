# Project: Martini Dyhedrals powered by AI

Based on [ProtTrans](https://github.com/agemagician/ProtTrans), we convert the output of the neural nextfwork to gest secondary structure probabilies and find the appropriate dyherals for Martini3 

## Requirement 

- Big GPU
- Memory (to define)



# Command line
```
pyhton -m venv my_virtual_env
source my_virtual_env/bin/activate
pip install -r requirements.txt

python run_ProtT5.py example/rcsb_pdb_1AO6.fasta
```

