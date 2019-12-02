from collections import Counter 
import numpy as np
from scipy.special import digamma, gammaln, polygamma
import warnings
warnings.filterwarnings("ignore")

def get_doc(file_name,stopwords_file):
    '''preprocess the real dataset to tokenize the words'''
    texts = []
    special_chars = '!"#$£@%&/()=?.,+-*\':;_´`1234567890'
    with open(file_name, 'r') as infile:
        copy = False
        text = ''
        for line in infile:
            if copy:
                if line.strip() == '</TEXT>':
                    text = text.lower()
                    texts.append(text)
                    text = ''
                    copy = False
                else:
                    for char in special_chars:
                        line = line.replace(char, '')
                    text += line
            else:
                if line.strip() == '<TEXT>':
                    copy = True
        tmp_texts = np.array(texts)
    
    stop_words_line = []
    with open(stopwords_file, 'r') as infile:
        data=infile.read().replace(',', ' ')
        for word in data.split():
            stop_words_line.append(word)
        
    stop_words = np.array(stop_words_line)
        
    corpus = []
    for text in tmp_texts:
        words = np.array(text.split())
            
        stopwords_filtered_document = [w for w in words if w not in stop_words]
        single_words = [k for k, v in Counter(stopwords_filtered_document).items() if v == 1 ]
        final_filtered_document = [w for w in stopwords_filtered_document if w not in single_words]
            
        if not final_filtered_document: # Document is empty, Shape = []
            continue
        corpus.append(final_filtered_document)
    return corpus







def initialize_parameters(corpus, voc, k, M):
    Phi = []
    gamma = np.zeros([M,k])
    alpha = np.ones([M,k])

    for m in range(0,M):
        doc = np.array(corpus[m])
        N = len(doc)
        phi = np.ones([N,k])/k
        gamma[m,:] = alpha [m,:] + N/k
        Phi.append(phi)
    # Initialize Beta
    Beta = np.random.uniform(0,1,(k,len(voc)))
    Beta = Beta/Beta.sum(axis=1).reshape(k,1)
    return Phi, gamma, alpha, Beta


def compute_likelihood(Phi, gamma, alpha, Beta, doc, voc, k):
    likelihood = 0.0
    V = len(voc)
    words = np.array(doc)
    N = len(words)
    
    alpha_sum = 0.0
    phi_gamma_sum = 0.0
    phi_lgb_sum = 0.0
    e_sum = 0.0
    gamma_sum = 0.0
    
    alpha_sum += gammaln(alpha.sum())  
    gamma_sum -= gammaln(gamma.sum()) 
    for i in range(0,k):
        #
        alpha_sum -= gammaln(alpha[i]) + (alpha[i] - 1) * (digamma(gamma[i]) - digamma(gamma.sum()))
        Phi_p= Phi[:,i] > 0
        w_ind = np.array(list(map(lambda x: np.sum(np.in1d(voc, x)),words[Phi_p])))   
        phi_gamma_sum = np.sum(Phi[Phi_p,i] * (digamma(gamma[i]) - digamma(gamma.sum())))
        e_sum = np.dot(Phi[Phi_p,i],np.log(Phi[Phi_p,i]))
        b_p=Beta[i,:]>0
        phi_lgb_sum += np.sum(np.outer((Phi[Phi_p,i] * w_ind), np.log(Beta[i,b_p])))
        gamma_sum += gammaln(gamma[i]) - (gamma[i] - 1) * (digamma(gamma[i]) - digamma(gamma.sum()))
    
    likelihood += (alpha_sum + phi_gamma_sum + phi_lgb_sum - gamma_sum - e_sum) 
    return likelihood


def E_step(Phi, gamma, alpha, Beta, corpus, voc, k, M):
    '''E-step: variational inference'''
    likelihood = 0.0
    #
    for d in range(0,M):
        words = np.array(corpus[d])
        N = len(words)
        phi = Phi[d]
        
        conv_counter = 0
        #
        while conv_counter < 100:
            
            phi_old = phi
            phi = np.zeros([N,k])
            gamma_old = gamma[d, :]
            
            for n in range(0,N):
                word = words[n]
                w_in_voc =np.where(voc == word)
                if len(w_in_voc[0]) > 0: # word exists in vocabulary
                    phi[n,:] = Beta[:,w_in_voc[0][0]]* np.exp(digamma(gamma[d,:]) - digamma(np.sum(gamma[d,:])))
                    phi[n,:] = phi[n,:] / np.sum(phi[n,:])   
            alpha = np.ones([M,k])
            gamma[d, :] = alpha[d, :] + np.sum(phi, axis=0)    
            
            conv_counter += 1
            # Check if gamma and phi converged
            if np.linalg.norm(phi - phi_old) < 1e-3 and np.linalg.norm(gamma[d,:] - gamma_old) < 1e-3:
                Phi[d] = phi               
                likelihood += compute_likelihood(Phi[d], gamma[d,:], alpha[d,:], Beta, corpus[d], voc, k)
                conv_counter=100
                
    return Phi, gamma, likelihood



def M_step(Phi, gamma, alpha, corpus, voc, k, M):
    V = len(voc)
    
    # 1 update Beta
    Beta = np.zeros([k,V])
    for d in range(0,M):
        words = np.array(corpus[d])
        voc_pos = np.array(list(map(lambda x: np.in1d(words, x),voc)))
        Beta += np.dot(voc_pos, Phi[d]).transpose()
    Beta = Beta / Beta.sum(axis=1).reshape(k,1)

    # 2 update alpha
    for i in range(1000):
    
        old_alpha = alpha
    # Calculate the gradient
        g = M*(digamma(np.sum(alpha))-digamma(alpha)) + np.sum(digamma(gamma)-np.tile(digamma(np.sum(gamma,axis=1)),(k,1)).T,axis=0)

    # Calculate Hessian 
        h = -M * polygamma(1,alpha)
        z = M * polygamma(1,np.sum(alpha))
    # Calculate parameter
        c = np.sum(g/h)/(1/z+np.sum(1/h))
    # Update alpha
        alpha -= (g-c)/h

        if np.sqrt(np.mean(np.square(alpha-old_alpha)))<1e-4:
            break

    return alpha, Beta
    
def variational_EM(Phi_init, gamma_init, alpha_init, Beta_init, corpus, voc, k, M):
    '''EM inplementation'''
    print('Variational EM')
    likelihood = 0
    likelihood_old = 0
    iteration = 1 # Initialization step is the first step
    Phi = Phi_init
    gamma = gamma_init
    alpha = alpha_init
    Beta = Beta_init
    while iteration <= 100 and (iteration <= 2 or np.abs((likelihood-likelihood_old)/likelihood_old) > 1e-4):
        # Update parameters 
        likelihood_old = likelihood
        Phi_old = Phi 
        gamma_old = gamma 
        alpha_old = alpha
        Beta_old = Beta
    
        Phi, gamma, likelihood = E_step(Phi_old, gamma_old, alpha_old, Beta_old, corpus, voc, k, M)
        alpha, Beta = M_step(Phi, gamma, alpha_old, corpus, voc, k, M)
                
    
        iteration += 1
        
    
        
    return Phi, gamma, alpha, Beta, likelihood
    
def inference_method(corpus, voc,k=2):
    '''use EM to do LDA'''
    M = len(corpus)   # nbr of documents
    Phi_init, gamma_init, alpha_init, Beta_init = initialize_parameters(corpus, voc, k, M)
    Phi, gamma, alpha, Beta, likelihood = variational_EM(Phi_init, gamma_init, alpha_init, Beta_init, corpus, voc, k, M)
    
    return Phi, gamma, alpha, Beta, likelihood