import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import boto3
import io
import haversine as hs
from geopy.distance import geodesic
import random as random
from inference_functions import load_image,get_labels,normalize_times, scale_up, scale_down
import yaml
import paho.mqtt.client as mqtt
from datetime import datetime 
import traceback
import argparse

#intended for better memory management at GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
	

def load_model_from_s3():
    bucket_name='w251-final-project-model'  
    #s3 = boto3.client('s3')
    
    
    print('Loading model from S3')
    model_dir='/tmp/inference_model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket(bucket_name)
    # download file into current directory
    for s3_object in my_bucket.objects.all():
        filename = s3_object.key
        my_bucket.download_file(s3_object.key,os.path.join(model_dir,filename))
    # model_file_name='saved_model.pb'
    # combined_pb_file=os.path.join(model_dir,model_file_name)
    # s3.download_file(bucket_name, model_file_name, combined_pb_file)
      
    tf.keras.backend.clear_session()
    #converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
    #model = converter.convert()
    model=tf.keras.models.load_model(model_dir)
    print('Loaded model from S3')
    return model


    
def setup():
    #get configurationvalues
    with open('inference.yml') as config_file:
        config_data = yaml.load(config_file)
    
    global model_dir
    model_dir = config_data['image_dir']
    global image_dir
    image_dir = config_data['image_dir']
    global latstart 
    latstart= config_data['latstart']
    global latend
    latend=config_data['latend']
    global longstart
    longstart = config_data['longstart']
    global longend
    longend=config_data['longend']
    global dtstart 
    dtstart= config_data['dtstart']
    global dtend
    dtend=config_data['dtend']
    
    print(config_data)
    
    #get model from s3
    #model=load_model_from_s3()

    #get model from local directory
    local_model_path=model_dir
    
    print ('Loading model', local_model_path)
    model=tf.keras.models.load_model(local_model_path,compile=False)
    print('Compiling model')
    model.compile(loss=hs.haversine_loss)
    print (model.summary())
    return (model)

def inference(image_array,file_name):
    #normalize values
    image_array=image_array/255
    
    #get labels to test image
    y_lat,y_long,time=get_labels(file_name)
    
    #process time into TF input
    T_test=normalize_times(time,dtstart,dtend)
    T_test=(T_test)
    T_test=np.expand_dims(T_test,axis=0)
    
    #process proper image dimensions
    X_test=np.expand_dims(image_array,axis=3)
    X_test=np.expand_dims(X_test,axis=0)
    
    #do prediction
    print('Model being called for prediction....')
    #print('X_test:',X_test[0:20])
    #print('t_test:',T_test)
    y_hat = model.predict([X_test,T_test])
    #print('y_hat:',y_hat)
    
    #output results
    y_hat_lat=y_hat[0,0]
    y_hat_long=y_hat[0,1]
        
    y_hat_lat=scale_up(y_hat_lat,latend,latstart)
    y_hat_long=scale_up(y_hat_long,longend,longstart)

    point1=(y_hat_lat,y_hat_long)
    point2=(y_lat,y_long)
    
    loss_nm=geodesic(point1,point2).nautical
    print('Estimated latitude, longitude:',y_hat_lat,',',y_hat_long)
    print('Actual latitude, longitude, time:',y_lat,',',y_long,time)
    print('Error in nautical miles:',loss_nm)
    print('\n------------------------------------------------------------------\n')
    
def on_log(mqttc, obj, level, string):
    print(string)
    return

def on_message(client,userdata, msg):
        
    try:
        #check if picture or filename
        if len(msg.payload)<100:
            print('Filename received!')
            global file_name
            file_name=msg.payload.decode()
            
        else:
            print("Celestial image received!",datetime.now(),'\n')
                
            #use numpy to construct an array from the bytes
            image_array = np.fromstring(msg.payload, dtype='uint8')
            reshaped_image=image_array.reshape(224,224)
            
            #do inference
            inference(reshaped_image,file_name)
        

    except:
        traceback.print_exc()
        quit(0)

def on_connect_local(client, userdata, flags, rc):
    print("connected to local broker with rc: " + str(rc))
    client.subscribe(LOCAL_MQTT_TOPIC,qos=1)
   
#loads model and gets ready for incoming picture
model=setup()

mqtt_flag=True

if mqtt_flag:
    #set up MQTT client
    LOCAL_MQTT_HOST="mosq-broker"
    LOCAL_MQTT_PORT=1883
    LOCAL_MQTT_TOPIC="celestial"

    #now connect and subscribe
    local_mqttclient = mqtt.Client()
    local_mqttclient.on_connect = on_connect_local
    local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 3600)
    #local_mqttclient.on_log = on_log

    #action if message
    local_mqttclient.on_message = on_message

    #loop keeps it listening
    local_mqttclient.loop_forever()

else:
    path, dirs, files = next(os.walk(image_dir))
    file_count = len(files)
    for i in range(10):
        random_file=files[random.randint(0,file_count-1)]
        print('Random file selected:', random_file)

        image_path=os.path.join(image_dir,random_file)    

        #doing image work here to reduce size of file sent on MQTT
        #read in image in black and white and convert size
        image = cv2.imread(image_path, 0) #0=bw
        dim=(224,224)
        image = cv2.resize(image, dim)
        print("Celestial image received!",datetime.now())
                    
        #use numpy to construct an array from the bytes
        #image_array = np.fromstring(msg.payload, dtype='uint8')
        #reshaped_image=image_array.reshape(224,224)
        
        #do inference
        inference(image,random_file)
            


