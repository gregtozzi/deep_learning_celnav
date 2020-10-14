#upload_model.py
#this isn't working right now as need a container to run tensorflow
#so just manually copying model by hand

import os
import boto3
import tensorflow as tf 

downloaded_model='/w251-project/inference/small_model'
model=tf.keras.models.load_model(downloaded_model)

model_dir='/tmp/inference_model'
model.save(model_dir)

def upload_model(model_dir,bucket_name):
    #s3_client=boto3.client('s3')
    session = boto3.Session()
    s3 = session.resource('s3')
    bucket = s3.Bucket(bucket_name)
    for subdir, dirs, files in os.walk(model_dir):
        for file in files:
            full_path = os.path.join(subdir, file)
            with open(full_path, 'rb') as data:
                bucket.put_object(Key=full_path[len(model_dir)+1:], Body=data)

bucket_name='w251-final-project-model'
upload_model(model_dir,bucket_name)