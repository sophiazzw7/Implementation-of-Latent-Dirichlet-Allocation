import sys
sys.path.append('../')

import numpy as np

from src.LDA_Gibbs import test_data
from src.LDA_EM import inference_method
from src.LDA_EM import initialize_parameters
from src.LDA_EM import compute_likelihood
from src.LDA_EM import E_step
from src.LDA_EM import M_step
from src.LDA_EM import variational_EM

    
documents, vocabulary=test_data(20)
vocabulary=np.array(vocabulary)

Phi, gamma, alpha, Beta, likelihood = inference_method(documents, vocabulary)

topk=2 #get top 2 key words
N_K = 2
for i in range(N_K):
	print("Top key words for topic",i+1)
	sorted_idx=np.argsort(-Beta[i], axis=0)

	for j in range(topk):
		aa=sorted_idx[j]
		print(vocabulary[aa])




