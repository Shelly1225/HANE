###
This repository provides a reference implementation of *HANE* as described in the [paper](DOI: 10.1109/TKDE.2021.3117274). 
#### **Required Packages**
* tensorflow
* numpy
* scipy
* scikit-learn
* networkx
* gensim (only for using DeepWalk as base embedding method)
* theano (only for using NetMF as base embedding method)


##### **dataset**
baiduyunpan：https://pan.baidu.com/s/1dD6TpleAUVq7AKk-lev1pw   (fnpe)

#### **How To Run**
Use `python main.py` to run the code with all the default settings. Here are some useful arguments that can be passed to the program:
* `--data`: name of the dataset (file located in `./dataset`), e.g., `--data /cora`.
* `--basic-embed`: name of the base embedding method, e.g., `--basic-embed deepwalk`.
* `--coarsen-level`: number of levels for coarsening, e.g., `--coarsen-level 2`.
* `--embed-dim`: dimensionality for embedding, e.g., `--embed-dim 128`.
* `--no-eval`: will not evaluate the embeddings if set (.truth file will not be required then).
* `--workers`: number of processes to run the code. 
* `--refine-type`: refinement method, including `MD-gcn` , `MD-dumb` (without model training), and `MD-gs` (using GraphSAGE).


## **If you find *HANE* useful in your research, please cite our paper:**
@ S. Zhao, Z. Du, J. Chen, Y. Zhang, J. Tang and P. Yu, "Hierarchical Representation Learning for Attributed Networks," in IEEE Transactions on Knowledge and Data Engineering, doi: 10.1109/TKDE.2021.3117274.
 
