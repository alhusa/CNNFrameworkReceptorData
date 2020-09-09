import pickle
import numpy as np

# Dict to give a vaule to each amino acid for onehot encoding.
amino_to_num = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'B': 4, 'C': 5, 'E': 6,
                    'Q': 7, 'Z': 8, 'G': 9, 'H': 10, 'I': 11, 'L': 12,
                    'K': 13, 'M': 14, 'F': 15, 'P': 16, 'S': 17, 'T': 18,
                    'W': 19, 'Y': 20, 'V': 21, '*': 22}

# num_to_amino = ['A', 'R', 'N', 'D', 'B', 'C', 'E',
#                     'Q', 'Z', 'G', 'H', 'I', 'L',
#                     'K', 'M', 'F', 'P', 'S', 'T',
#                     'W', 'Y', 'V', '*']


# Loads the data augmented by a function in ImmuneML.
filename = '/Users/ahusa/Documents/bio_master/master_thesis/data/from_immuneML/real_individual.pickle'
sequences = pickle.load(open(filename, 'rb'))

# Array to store data
data = np.zeros((len(sequences), 1, 23, 25))
label = np.zeros(len(sequences))

# One hot encodes the data
for i in range(len(sequences)):
    j = 0
    for char in sequences[i]:
        if char == '0' or char == '1':
            label[i] = int(char)
            break
        data[i, :, amino_to_num[char], j] = 1
        j += 1

# Shuffle the data.
indices = list(range(len(label)))
np.random.shuffle(indices)

data = data[indices, :, :, :]
label = label[indices]


# Stores the one hot encoded data as a pickle file.
with open('/Users/ahusa/Documents/bio_master/master_thesis/data/onehot_encoded/onehot_real_individual.pickle', 'wb') as file:
    pickle.dump(data, file)
    pickle.dump(label, file)
