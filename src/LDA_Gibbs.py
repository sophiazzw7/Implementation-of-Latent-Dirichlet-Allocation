from collections import Counter 
import numpy as np
from scipy.special import digamma, gammaln


def pos_in_voc(corpus, voc):
    '''generate a matrix to store each word's position in the dictionary '''
    loc_doc=np.copy(corpus)
    for i in range(len(corpus)):
        for j in range(len(corpus[i])):
            idx=np.where(voc==corpus[i][j])[0]
            if len(idx)==0:
                voc=np.append(voc,corpus[i][j]) 
                idx=np.where(voc==corpus[i][j])[0]
            loc_doc[i][j]=idx[0]
    return loc_doc, voc

def pre_processing(corpus,voc, N_K):
    # wt: a word-topic count matrix
    # Z: word-topic assignment matrix
    # dt: document-topic count matrix
    # n_z : the total number of words assigned to topic z
    # n_m: the total number of words in each document in corpus
    Z = []
    M=len(corpus)
    V=len(voc)  #number of words in vocabulary
    dt = np.zeros([M, N_K])
    #word in the topic
    wt = np.zeros([N_K, V])
    n_z = np.zeros(N_K)
    n_m = []
    
    # randomly assign topics to words in each doc
    for m in range(M):
        z_n = []
        doc=corpus[m]
        N=len(doc)
        for n in range(N):
            t=doc[n]
            z = np.random.randint(0, N_K)
            z_n.append(z)
            dt[m, z] += 1
            wt[z, t] += 1
            n_z[z] += 1
        Z.append(np.array(z_n))
        n_m.append(len(corpus[m]))
        
    return Z, dt, wt, n_z, np.array(n_m), V


def initialize(N_D, N_W, N_K, doc):
    # Dirichlet priors
    alpha = 1
    gamma = 0.5

    np.random.seed(115)
    # Z := word topic assignment
    Z = np.random.randint(N_K,size=(N_D, N_W))  # randomly assign word's topic

    # Pi := document topic distribution
    Pi = np.random.dirichlet(alpha*np.ones(N_K),size=N_D)

    # B := word topic distribution
    B = np.random.dirichlet(gamma*np.ones(N_W),size=N_K)
    
    return alpha,gamma, Pi, B


def sample_conditionally(loc_doc,doc,doc_word, N_K, lenV, n_z, n_m, alpha, phi, Z, dt, wt):
    z0=Z[doc][doc_word]  #initial topic assignment to word
    pos=loc_doc[doc][doc_word] # position of the word in voc
    
    #exclude the word
    dt[doc][z0]=dt[doc][z0]-1
    wt[z0,pos]=wt[z0,pos]-1
    
    # resample topic for current word
    p_z = ((wt[:,pos] + phi)/(n_z + lenV * phi)) * ((dt[doc,:] + alpha)/(n_m[doc] + N_K * alpha))

    p_z=p_z/p_z.sum()
    z1 = np.random.multinomial(1, p_z).argmax()
    Z[doc][doc_word]= z1
    
    # add the word back
    dt[doc][z1]=dt[doc][z1]+1
    wt[z1,pos]=wt[z1,pos]+1
    
    return Z, dt, wt


def lda_gibbs(loc_doc,iteration, N_K, lenV, n_z, n_m, alpha, gamma, phi, Z, dt, wt, N_D, B, Pi):
    """Gibbs Sampler in LDA"""
    np.random.seed(116)
    beta_sum=[]
    theta_sum=[]

    for i in range(iteration):
        for j in range(len(loc_doc)):
            for k in range(len(loc_doc[j])):
            
                Z, dt, wt = sample_conditionally(loc_doc,j,k, N_K, lenV, n_z, n_m, alpha, phi, Z, dt, wt)

        
        temp_beta=(wt + alpha)/(n_z[:,None] + lenV *alpha)
        beta_sum.append(temp_beta)
        temp_theta = (dt + Pi)/(n_m[:, None] + N_K * Pi)
        theta_sum.append(temp_theta)
        
        
        
    return beta_sum, theta_sum, wt


def get_doc(file_name,stopwords_file):
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
        
    documents = []
    for text in tmp_texts:
        words = np.array(text.split())
            
        stopwords_filtered_document = [w for w in words if w not in stop_words]
        single_words = [k for k, v in Counter(stopwords_filtered_document).items() if v == 1 ]
        final_filtered_document = [w for w in stopwords_filtered_document if w not in single_words]
        ys = []
        for x in final_filtered_document:
            ys.append(x)

        final_filtered_document=ys
        if not final_filtered_document: # Document is empty, Shape = []
            continue
        documents.append(final_filtered_document)
    print("The length of the corpus is: ", len(documents))
    return documents


def test_data(docnum = 200):
    """Generating test data."""
    def generating(words, probs):
        '''generate words based on given probabilities'''
        wordslist = []; all_theta = []
        c = (0.4,0.5)
        theta = np.random.dirichlet(c)
        all_theta.append(theta)
        N = np.random.poisson(15)  
        for i in range(0,N):
            # Draw topics from multinomial distribution
            z_n = np.random.multinomial(1,theta)
            z_n = np.nonzero(z_n)[0][0]
            w_n = np.random.multinomial(1,probability[z_n,:])
            w_n = np.nonzero(w_n)[0][0]
            wordslist.append(words[z_n][w_n])
        return wordslist, all_theta
    # generate vocabulary
    voc = ['bayesian','probability','posterior','prior',
            'add','minus','multiply','divide']
    #Initializing words and topics
    words = []
    words.append(['bayesian','probability','posterior','prior'])
    words.append(['add','minus','multiply','divide'])
    probability = np.array(([0.25,0.25,0.25,0.25],
                           [0.7,0.05,0.05,0.2]))
    
    docs = []; doci = []
    for i in range(0, docnum):
    	doci,theta = generating(words, probability)
    	docs.append(doci)
    return docs, voc

