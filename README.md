# Description
This repository contains the code and datasets to reproduce the results and figures and to train the models from our paper "The substrate scopes of enzymes: a general prediction model based on machine and deep learning".


#### For people interested in using the trained prediction model, we implemented a [web server](https://esp.cs.hhu.de/) that allows an easy use of our trained model. The prediction tool can be run in a web-browser and does not require the installation of any software. Prediction results are usually ready within a few minutes. Example inputs can be found on the homepage.

#### For people interested in using a python function to achieve predictions of the trained model, we created a [GitHub repository](https://github.com/AlexanderKroll/ESP_prediction_function) that allows an easy use of our trained model.

## Downloading data folder
Before you can run all scripts of this repository, you need to [download and unzip an additional data folder from Zenodo](https://doi.org/10.5281/zenodo.8016269).
Afterwards, this repository should have the following strcuture:

    ├── notebooks_and_code
    ├── data
    ├── additional_data_ESP            
    └── README.md

## Using code and reporducing results
All code to reproduce the results is available in the form of Jupyter Notebooks in the folder "notebooks_and_code". All code and produced output files are available in the folder "data".

## Requirements for running the code in this GitHub repository
The code was implemented and tested on Windows with the following packages and versions (installation took ~20 minutes)
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
