#!/bin/sh

#s3fs rza-cos-standard-l5d /data/stars/images -o url=https://s3.us-east.cloud-object-storage.appdomain.cloud -o passwd_file=/data/credentials_file
#s3fs -o nonempty rza-cos-standard-l5d /data/stars/images -o url=https://s3.us-east.cloud-object-storage.appdomain.cloud -o passwd_file=/data/credentials_file

#google-drive-ocamlfuse /w251-project/stars
python3 get_skies_random.py #does a random selection of set # of locations
#python3 get_arbitrary_skies.py [does a grid search]

