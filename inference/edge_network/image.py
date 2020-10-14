import numpy as np
import cv2
import paho.mqtt.client as mqtt
import yaml
import os
import random
import time

MQTT_HOST="mosq-broker"
MQTT_PORT=1883
MQTT_TOPIC="celestial"
mqttclient = mqtt.Client()
mqttclient.connect(MQTT_HOST, MQTT_PORT, 3600)

with open('/inference/inference.yml') as config_file:
        config_data = yaml.load(config_file)
        image_dir = config_data['image_dir']

#collect images
path, dirs, files = next(os.walk(image_dir))
file_count = len(files)

while (True):
    random_file=files[random.randint(0,file_count-1)]
    print('Random file selected:', random_file)

    image_path=os.path.join(image_dir,random_file)    

    #doing image work here to reduce size of file sent on MQTT
    #read in image in black and white and convert size
    image = cv2.imread(image_path, 0) #0=bw
    dim=(224,224)
    image = cv2.resize(image, dim)
    
    #sending filename first
    msg=random_file
    mqttclient.publish(MQTT_TOPIC, payload=msg, qos=1)
    print('Sent filename to broker')

    #now sending image
    msg=bytearray(image)
    mqttclient.publish(MQTT_TOPIC, payload=msg, qos=1)
    print('Sent message to broker')

    
    imS = cv2.resize(image, (960, 540))                    # Resize image
    cv2.imshow("sky", imS) 
    k=cv2.waitKey()

    if k==27:  ##hit esc key to break from program, otherwise keypress cycles to next image
        break
    

#need program to keep running to make sure it hits broker once
  
  
  
 
  