#######################################################
# use real data to test LDA EM model
# dataset is based on the 2246 documents from the Associated Press
# The original dataset is available at:
# https://github.com/blei-lab/lda-c/tree/master/example
#######################################################
import sys
sys.path.append('../')

from src.LDA_Gibbs import test_data
from src.LDA_EM import inference_method
from src.LDA_EM import initialize_parameters
from src.LDA_EM import compute_likelihood
from src.LDA_EM import E_step
from src.LDA_EM import M_step
from src.LDA_EM import variational_EM
from src.LDA_Gibbs import get_doc 

import numpy as np

#######################################################
# Read in the original dataset with 2204 documents
# Attention!!!!!!!!: this will take hours to run
#######################################################
documents = get_doc('../data/ap.txt',  '../data/stopwords.txt')

#######################################################
# Since it takes hours to run the whole data set, here 
# only use 6 documents from the original
# dataset as a demonstration
#######################################################
documents = get_doc('../data/sample1.txt',  '../data/stopwords.txt')

vocabulary=np.genfromtxt('../data/vocab.txt',  dtype='str')
N_K = 3
Phi, gamma, alpha, Beta, likelihood = inference_method(documents, vocabulary,N_K)

topk=3 #get top 3 key words
for i in range(N_K):
	print("Top key words for topic",i+1)
	sorted_idx=np.argsort(-Beta[i], axis=0)

	for j in range(topk):
		aa=sorted_idx[j]
		print(vocabulary[aa])