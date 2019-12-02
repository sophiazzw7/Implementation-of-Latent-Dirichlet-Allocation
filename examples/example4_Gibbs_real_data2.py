#######################################################
# This data set contains several public files of religious and spiritual texts. Also included is a “wildcard” file on the subject 
# of machine super intelligence. This file is licensed under a Creative Commons Attribution-Share Alike 2.5 Switzerland License. 
# More information can be found at: https://creativecommons.org/licenses/by-sa/2.5/ch/deed.en. 
# The original dataset is available at:https://www.kaggle.com/metron/public-files-of-religious-and-spiritual-texts
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

documents = get_doc('../data/real_data.txt',  '../data/stopwords.txt')
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

