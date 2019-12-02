# Implementation of Latent Dirichlet Allocation using Python
Latent Dirichlet Allocation (LDA) is a very important model in machine learning area, which can
automatically extract the hidden topics within a huge amount of documents and further represent
the theme of each document as an ensemble of topics. It is a three-level hierarchical Bayesian
model, in which each item of a collection is modeled as a finite mixture over an underlying set
of topics. Each topic is, in turn, modeled as an infinite mixture over an underlying set of topic probabilities. In the context of text modeling, the topic probabilities provide an explicit representation of a document. Due to the intractability of LDA posterior inference, many approximate inference techniques have been invented.In this study, Two efficient approximate inference techniques are used here: EM algorithm and gibbs algorithm, for empirical Bayes parameter estimation.

## Development Environment
* Language: Python3
## Environment Setup
* Fetch git repo:
```shell
git clone https://github.com/sophiazzw7/Implementation-of-Latent-Dirichlet-Allocation.git
cd Implementation-of-Latent-Dirichlet-Allocation
```
* Install packages:
```shell
pip install pip install LDA-project-19
```
* Use packages:
```shell
from src.LDA_Gibbs import function_name
Note: here the function_name can be the the following functions: pos_in_voc, pre_processing, initialize, 
sample_conditionally, lda_gibbs,  get_doc, test_data

from src.LDA_EM import function_name
Note: here the function_name can be the following functions:initialize_parameters, compute_likelihood, E_step, 
M_step, variational_EM, inference_method
```


## Example of test data
To test the algorithm, we created a function to generate a test dataset with known parameters for testing purpose.

## Example of real datasets
 Associated Press Data

We applied our LDA models to the real dataset. The dataset is based on the 2204 documents from the Associated Press. The original dataset is available at: https://github.com/blei-lab/lda-c/tree/master/example


## References
* [Latent Dirichlet Allocation](http://www.cs.columbia.edu/~blei/papers/BleiNgJordan2003.pdf)

## Special thanks to:
Mingjie Zhao, for her contributions while building the algorithm.


