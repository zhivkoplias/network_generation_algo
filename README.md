FFL-based preferential attachment algorithm: TF-network graph model
================
Erik Zhivkoplias, Oleg Vavulov, Thomas Hillerton

August 13, 2021



## Introduction

Gene-gene and gene-protein regulatory relationships form a gene regulatory network (GRN) that controls the cellular response to changes in the environment. A number of inference methods to reverse engineer the original GRN from large-scale expression data have recently been developed. However, the absence of ground-truth networks when evaluating the performance makes realistic simulations necessary. Existing in-silico data generators however do not always capture all properties of known biological networks. 

Here we present a new algorithm, FFLatt, for building GRN graphs with controlled topological parameters, developed based on properties of four experimentally derived GRNs. It aims to contribute to more accurate and robust performance evaluation of existing reverse-engineering network inference methods. The novelty of the presented algorithm is that it generates networks with boosted feed-forward loop motif (FFL), known to be important for network dynamics.

## Download and install Conda

Download and install the latest Anaconda / Miniconda for your operation system. Example for Linux is provided below:

``` bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source .bashrc
```

### Clone repo and install conda environment

First clone this github repo:

``` bash
git clone https://github.com/zhivkoplias/network_generation_algo
cd network_generation_algo/envs
```

Now proceed with conda environment installation:

``` bash
conda create --name build_ffl_env --file build_ffl_env.yml
conda activate build_ffl_env

```


## Usage

Main scripts are located in ```src``` folder
To test the algorithms use one of the test scripts provided in ```snippets``` folder

 - test.py
 - parameter_space_exploration.py
 - Figure5.ipynb etc.

Good luck!


