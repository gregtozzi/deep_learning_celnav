# Dockerfile for the image preprocessor.

FROM ubuntu

# Prevents questions from hanging the build
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
                apt-get update && \
		apt-get install -y python3-pip && \
		apt-get install -y libglib2.0-0 && \
                pip3 install PyYAML && \
		pip3 install numpy && \
                apt-get install -y libsm6 libxext6 libxrender-dev && \
		pip3 install opencv-python
		
		
ADD preprocessor.py /
ADD preprocessor.yml /

