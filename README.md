FFL-based preferential attachment algorithm: TF-network graph model
================
Erik Zhivkoplias, Oleg Vavulov, Thomas Hillerton

August 13, 2021



## Introduction

TBD

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
 - artificial_network_generation.ipynb

Good luck!


