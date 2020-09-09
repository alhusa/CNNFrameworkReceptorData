import numpy as np
import csv
import pandas as pd



# Generates random seqeunces and stores them in a VDJdb format.



num_to_amino = list("ACDEFGHIKLMNPQRSTVWY")
num_to_amino.sort()



n_samples = 10000
seq_length = 18
id = list(np.arange(n_samples) + 1)
# Array to store data
data = np.random.randint(0,len(num_to_amino),(n_samples, seq_length), dtype='int')
aminoacid_seq = []
for i in range(n_samples):
    seq = ''
    for j in range(seq_length):
        seq += num_to_amino[data[i][j]]

    aminoacid_seq.append(seq)

df = pd.DataFrame(list(zip(id,aminoacid_seq)),columns=['complex.id','CDR3'])

df = df.assign(**{'Species': None, 'MHC A': None, 'MHC B': None, 'MHC class': None,
                 'Reference': None, 'Method': None, 'Meta': None, 'CDR3fix': None, 'Score': None,
                  'V': 'no_data', 'Gene': 'TRB', 'J':'no_data', 'Epitope':'no_data', 'Epitope gene':'no_data',
                  'Epitope species':'no_data',})
df = df[['complex.id', 'Gene', 'CDR3', 'V', 'J', 'Species', 'MHC A', 'MHC B',
       'MHC class', 'Epitope', 'Epitope gene', 'Epitope species', 'Reference',
       'Method', 'Meta', 'CDR3fix', 'Score']]


df.to_csv('/Users/ahusa/Documents/bio_master/immuneml_folder/VDJDataset/onehottest/generated_test_10.tsv', sep='\t', index=False)
