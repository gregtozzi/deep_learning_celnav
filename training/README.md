[_Return to the white paper_](https://github.com/travisrmetz/w251-project#Train)

# Model Training in the Cloud
## W251 - Final Project - Tozzi/Metz - Summer 2020
### Setting up Container for Training

This directory contains a dockerfile to build the training container, a python script to include in the container, and a YAML file to provide user input to the training process.

The dockerfile is built on `tensorflow/tensorflow:latest-gpu-jupyter`.  It adds `pandas`, `ktrain`, and `h5py`.

#### Build Docker container

`docker build -t train -f training.dockerfile .`

#### Train a model

`nvidia-docker run -dit --name train -v /root/sets:/data/sets -v/root/models:/data/models tensorflow/tensorflow:latest-gpu-jupyter python3 /training.py`
