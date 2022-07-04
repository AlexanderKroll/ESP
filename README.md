# Description
This repository contains the code and datasets to reproduce the results and figures and to train the models from our paper "The substrate scopes of enzymes: a general prediction model based on machine and deep learning".


#### For people interested in using the trained prediction model, we implemented a [web server](https://esp.cs.hhu.de/) that allows an easy use of our trained model. The prediction tool can be run in a web-browser and does not require the installation of any software. Prediction results are usually ready within a few minutes.


## Requirements for running the code in this GitHub repository

- python 3.7.7
- jupyter
- pandas 1.3.0
- torch 1.6.0
- numpy 1.21.2
- rdkit 2020.03.3
- fair-esm 0.3.1
- py-xgboost 1.2.0
- matplotlib 3.4.1
- hyperopt 0.25
- sklearn 0.22.1
- pickle
- Bio 1.78
- re 2.2.1

The listed packaged can be installed using conda and pip:

```bash
pip install torch
pip install numpy
pip install tensorflow
pip install fair-esm
pip install jupyter
pip install matplotlib
pip install hyperopt
pip install pickle
pip install biopython
conda install pandas=1.3.0
conda install -c conda-forge py-xgboost=1.2.0
conda install -c rdkit rdkit
```
