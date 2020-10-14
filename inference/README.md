[_Return to the white paper_](https://github.com/gregtozzi/deep_learning_celnav#Edge)

# Celestial Inference at the Edge
## W251 - Final Project - Tozzi/Metz - Summer 2020
### Setting up Containers on Jetson for Inference

This section has files for setting up container on Jetson that will subscribe to a MQTT broker and 'listen' for images being published by the 'camera' container (described in /inference/edge_network).  Upon receiving the images, it uses the trained model to predict latitude and longitude.  The camera container also sends the name of the file, which has the time the synthetic picture was generated (which is an input to model along with the image) and the true latitude and longitude, which is parsed to provide an evaluation of accuracy after a prediction is made.

#### Download model

The model we used for the Jetson (with approximately 12m parameters) is [stored here.](https://bobby-digital.s3.us-east.cloud-object-storage.appdomain.cloud/model_for_travis.h5)  Download and store in `/inference/model` folder.


#### Build Docker containers with TF2, CV2 etc

`sudo docker build -t inference-image -f Dockerfile.inference .`

(If storing a model in an AWS bucket, which we built flexibility for, have to have credentials and config file for AWS in folder where Dockerfile is - copies them to container root.  This is for object store model.  Currently easiest way to install model is download per above and store locally.)


#### Start inference container and run inference

```docker run --name inference --memory="8g" --memory-swap="16g" -v /tmp:/tmp -v /w251-project/inference/:/inference/ --network project --runtime nvidia --privileged -ti --rm -v /tmp/.X11-unix/:/tmp/.X11-unix:rw -e DISPLAY=$DISPLAY inference-image bash```

Run inference from within container: `bash inference.sh` (this also runs Jetson clocks and does some buffer work to try and optimize for memory and performance)

#### In action

![Camera, broker and inference engine at work](https://github.com/travisrmetz/w251-project/blob/master/report_images/screenshot_of_inference.png)



