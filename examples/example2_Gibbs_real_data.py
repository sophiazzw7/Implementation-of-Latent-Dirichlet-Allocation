#######################################################
# use real data to test LDA Gibbs model
# dataset is based on the 2204 documents from the Associated Press
# The original dataset is available at:
# https://github.com/blei-lab/lda-c/tree/master/example
#######################################################
import sys
sys.path.append('../')


from src.LDA_Gibbs import pos_in_voc
from src.LDA_Gibbs import pre_processing
from src.LDA_Gibbs import initialize
from src.LDA_Gibbs import sample_conditionally
from src.LDA_Gibbs import lda_gibbs 
from src.LDA_Gibbs import get_doc 
from src.LDA_Gibbs import test_data


import numpy as np
np.random.seed(116)
#######################################################
# Read in the original dataset with 2204 documents
# Attention!!!!!!!!!: this will take hours to run
#######################################################
documents = get_doc('../data/ap.txt',  '../data/stopwords.txt')

#######################################################
# Since it takes hours to run the whole data set, here 
# only use 6 documents from the original
# dataset as a demonstration
#######################################################
documents = get_doc('../data/sample1.txt',  '../data/stopwords.txt')
W=np.genfromtxt('../data/vocab.txt',  dtype='str')


loc_doc, vocabulary=pos_in_voc(documents, W)   
phi=0.2
N_D = len(documents)  # num of docs

N_K = 3  # num of topics
max_iter=1000  # number of gibbs sampling iterations


Z, dt, wt, n_z, n_m, lenV=pre_processing(loc_doc,vocabulary, N_K)

alpha,gamma, Pi, B=initialize(N_D, lenV, N_K, loc_doc)
beta_sum,  theta_sum, wtt=lda_gibbs(loc_doc, max_iter, N_K, lenV, n_z, n_m, alpha, gamma, phi, Z, dt, wt, N_D, B, Pi)    

Beta=beta_sum[0]
Theta=theta_sum[0]

Beta=np.array(beta_sum).mean(axis=0)
print("Beta\n",Beta)

### print Theta
for i in range(1,max_iter):
	for j in range(len(Theta)):
		Theta[j]=Theta[j]+theta_sum[i][j]

for i in range(len(Theta)):
	Theta[i]=Theta[i]/max_iter
print("Theta\n",Theta)

