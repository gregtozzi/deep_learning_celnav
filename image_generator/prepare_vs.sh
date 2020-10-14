#!/bin/sh

# Create a mountpoint and link to it
mkdir <mountpoint>
s3fs <bucket> <mountpoint> -o url=https:<endpoint> -o passwd_file=credentials

# Build the Docker image
docker build -t imgen -f imgen.dockerfile .