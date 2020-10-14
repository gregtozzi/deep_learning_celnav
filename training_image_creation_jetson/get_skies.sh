#!/bin/sh

#first delete all images on local drive and up in S3
python3 delete_images.py

#should allow for google drive to also populate for use on colab
#maybe have to mount with -o nonempty
#google-drive-ocamlfuse /w251-project/stars

#then generate random images from box defined by ssc_generatory.yml
python3 get_skies_random.py 

#uploads to s3 bucket
#if stellarium does not quit properly then this needs to be run by hand
python3 upload_images_s3.py



#OLD AND NOT USED
#s3fs rza-cos-standard-l5d /data/stars/images -o url=https://s3.us-east.cloud-object-storage.appdomain.cloud -o passwd_file=/data/credentials_file
#s3fs -o nonempty rza-cos-standard-l5d /data/stars/images -o url=https://s3.us-east.cloud-object-storage.appdomain.cloud -o passwd_file=/data/credentials_file
#google-drive-ocamlfuse /w251-project/stars
#if doing grid
#python3 get_arbitrary_skies.py

