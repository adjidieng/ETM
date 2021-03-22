from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import random
from scipy import sparse
import itertools
from scipy.io import savemat, loadmat
from pathlib import Path

# Maximum / minimum document frequency
max_df = 0.7
min_df = 100  # choose desired value for min_df


# Read data
print('reading text file...')
data_file = '../balobi_nini/data/cleanned_tweets_2021_text_only.txt'
with open(data_file, 'r') as f:
    docs = f.readlines()

# Create count vectorizer
print('counting document frequency of words...')
vectorizer = TfidfVectorizer(lowercase=True, 
                             strip_accents="unicode", 
                             ngram_range = (1, 2),
                             max_features=200000) 
doc_terms_matrix = vectorizer.fit_transform(docs)
terms = vectorizer.get_feature_names()

# Get vocabulary
print('building the vocabulary...')

v_size = len(terms)
vocabulary = terms
print('  initial vocabulary size: {}'.format(v_size))

# Create bow representation
print('creating bow representation...')



# Save vocabulary to file
path_save = Path.cwd().joinpath('data', 'preprocess')
Path(path_save).mkdir(exist_ok=True, parents=True)

with open(path_save.joinpath('vocab.pkl'), 'wb') as f:
    pickle.dump(vocabulary, f)
del vocabulary

# Split bow intro token/value pairs
print('splitting bow intro token/value pairs and saving to disk...')

savemat(path_save.joinpath('tf_idf_doc_terms_matrix'), {"doc_terms_matrix": doc_terms_matrix}, do_compression=True)
savemat(path_save.joinpath('tf_idf_terms'), {"terms" : terms}, do_compression=True)

print('**' * 10 , doc_terms_matrix.shape)

print('Data ready !!')
print('*************')

