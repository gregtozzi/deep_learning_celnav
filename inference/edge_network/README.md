[_Return to the white paper_](https://github.com/gregtozzi/deep_learning_celnav#Edge)

# Celestial Inference at the Edge
## W251 - Final Project - Tozzi/Metz - Summer 2020
### Setting up Containers on Jetson for Image Capture and MQTT Image Transfer

This folder contains files for setting up a group of containers on a Jetson.  It is intended to replicate an edge device that estimates a latitude and longitude based on a camera's picture of the sky.  In our case it we are using random synthetic images rather than an actual camera.

We set up two containers.
1.  A camera container (`send_image`) that takes images generated using the functions of `test_images` and publishes them to a MQTT broker.
2.  A MQTT broker (`mosq-broker`) that receives messages and passes them along.

A third container (`inference`) is described in the README at \inference.  It subscribes to the `mosq-broker`, which means it is sent images from the `send_image` container.

In simulation mode, the camera container send a new image to the broker (and thus on to the inference container upon a key click.  The esc button is used to exit the send_image program.

Link to a demo video:  https://zoom.us/rec/play/vcEvdr39r2g3SNPDtASDV_UtW421e_-s1SUf-PYOnU-zACRQMVqmMOZGNLBaIGprp4ITygeePJec76GT



#### create jetson network for various containers
docker network create project

#### Set up and run MQTT broker container
```docker build -t broker-image -f Dockerfile.broker .```

```docker run --name mosq-broker --rm -p 1883:1883 -v /w251-project/inference:/inference --network project -ti broker-image mosquitto -v```


#### Set up and run virtual camera container

```docker build -t send-image -f Dockerfile.send .```

Before running container: `xhost +` (provides access to display)

From within edge_network folder
```docker run -e DISPLAY=$DISPLAY --privileged  --name send_image -v /w251-project/inference:/inference --rm -v /tmp/.X11-unix:/tmp/.X11-unix --network project -ti send-image bash```

From within container `bash image.sh`







