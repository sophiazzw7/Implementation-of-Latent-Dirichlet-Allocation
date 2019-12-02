#######################################################
# This data set contains several public files of religious and spiritual texts. Also included is a “wildcard” file on the subject 
# of machine super intelligence. This file is licensed under a Creative Commons Attribution-Share Alike 2.5 Switzerland License. 
# More information can be found at: https://creativecommons.org/licenses/by-sa/2.5/ch/deed.en. 
# The original dataset is available at:https://www.kaggle.com/metron/public-files-of-religious-and-spiritual-texts
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


documents = get_doc('../data/real_data.txt',  '../data/stopwords.txt')

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