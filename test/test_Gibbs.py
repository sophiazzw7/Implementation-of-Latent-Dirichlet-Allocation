
import sys
sys.path.append('../')

from src.LDA_Gibbs import pos_in_voc
from src.LDA_Gibbs import pre_processing
from src.LDA_Gibbs import initialize
from src.LDA_Gibbs import sample_conditionally
from src.LDA_Gibbs import lda_gibbs 
from src.LDA_Gibbs import test_data


import numpy as np

#######################################################
# use function to generate test data
#######################################################
documents, vocabulary=test_data(20)
vocabulary=np.array(vocabulary)


loc_doc, vocabulary=pos_in_voc(documents, vocabulary)


phi=0.2
N_D = len(documents)  # num of docs
N_K = 2  # num of topics
max_iter=5000  # number of gibbs sampling iterations


Z, dt, wt, n_z, n_m, lenV=pre_processing(loc_doc,vocabulary, N_K)
alpha,gamma, Pi, B=initialize(N_D, lenV, N_K, loc_doc)
beta_sum,  theta_sum, wtt=lda_gibbs(loc_doc, max_iter, N_K, lenV, n_z, n_m, alpha, gamma, phi, Z, dt, wt, N_D, B, Pi)    

# topk=2 #get top 2 key words
Beta=beta_sum[0]
Theta=theta_sum[0]


### print Beta
for i in range(1, max_iter):
	for j in range(N_K):
		Beta[j]= [Beta[j][q] + beta_sum[i][j][q] for q in range(len(vocabulary))]


for i in range(len(Beta)):
	Beta[i]=Beta[i]/max_iter
print("Beta\n",Beta)

### print Theta

for i in range(1,max_iter):
	for j in range(len(Theta)):
		Theta[j]=Theta[j]+theta_sum[i][j]

for i in range(len(Theta)):
	Theta[i]=Theta[i]/max_iter
print("Theta\n",Theta)





