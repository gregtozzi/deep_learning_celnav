# Dockerfile for model training.

FROM tensorflow/tensorflow:latest-gpu-jupyter

ARG DEBIAN_FRONTEND=noninteractive

RUN pip3 install ktrain && \
    pip3 install pandas && \
    pip3 install -q pyyaml h5py
    
ADD training.py /
ADD training.yml /
