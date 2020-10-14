# Toward Automated Celestial Navigation with Deep Learning

#### [Greg Tozzi](https://www.linkedin.com/in/gregorytozzi/) | August, 2020

*This is my final project—completed with [Travis Metz](https://www.linkedin.com/in/travis-metz-692985/)—from the [deep learning in the cloud and at the edge](https://www.ischool.berkeley.edu/courses/datasci/251) course that I took as part of the UC Berkeley School of Information's [Master of Information and Data Science](https://datascience.berkeley.edu) program.  We examined the feasiblity of automated deep learning-enabled celestial navigation by training a hybrid deep and convolutional neural network on synthetic images of the sky coupled with a representation of the time at which the images were taken.  We implemented the haversine formula as a custom loss function for the model.  We generated, pre-processed, and trained on synthetic images in the cloud and conducted inferencing on a Jetson TX2.*


**Skills demonstrated:** *Deep regression for computer vision* | *Implementing custom loss functions* | *Training in the cloud* | Generating synthetic data | Deploying inferencing to the edge | Containerizing application components with Docker

**Languages and frameworks**: *Python* | *TensorFlow* | *Keras* | *ktrain* | *Shell scripting*

### Continue on to read the white paper

### [Back to Portfolio](https://github.com/gregtozzi/portfolio)

##  <a id="Contents">Contents
[1.0 Introduction](#Introduction)

[2.0 System Overview](#System_Overview)

[3.0 The Model](#Model)

[4.0 Generating Synthetic Images](#Imgen)

[5.0 Training the Model](#Train)

[6.0 Inferencing at the Edge](#Edge)

[7.0 An End-to-End Example](#End)

## <a id="Introduction">1.0 Introduction

### 1.1 A _Very_ Brief Overview of Marine Navigation
Navigation, along with seamanship and collision avoidance, is one of the mariner's fundamental skills.  Celestial methods were once the cornerstone of maritime and aeronautical navigation, but they have been almost entirely supplanted by electronic means—first by terrestrially-based systems like OMEGA [[1]](#1) and LORAN [[2]](#2), then by satellite-based systems like GPS [[3]](#3), and then by mixed systems like differential GPS (DGPS) [[4]](#4).   Unfortunately, electronic navigation is fragile both to budgetary pressure and to malicious acts [[5]](#5)[[6]](#6).  Building and maintaining proficiency in celestial navigation is difficult, however, and a skilled navigator is still only able to fix a vessel's position a few times daily, weather permitting [[7]](#7).

In current practice, marine celestial navigation requires:

- A clear view of the horizon.  The navigator measures angles of elevation with a sextant using the horizon as a baseline.
- A clear view of celestial bodies.  Two bodies are required to fix the vessel's position, but three are typically used in practice.  Repeated observations of the sun can be combined to provided what is referred to as a _running fix_ of a vessel's position.
- An accurate clock.  Without a clock, a navigator can only compute the vessel's latitude.
- A _dead reckoning_ plot on which the navigator estimates the ship's track.
- For manual computation, a number of volumes containing published tables.

An automated hybrid inertial-celestial navigation system for aircraft, referred to as _astroinertial_ navigation, exists aboard high-end military aircraft and missile systems.  These systems rely on catalogs of stars and an estimated position developed through integration of accelerometer data [[8]](#7).

### 1.2 Intent of this Project
We want to explore the possibility of applying deep learning to the task of automating celestial  navigation.  We entered this project wanting to answer the following questions:

- Can we demonstrate the possibility of a viable solution to marine celestial navigation by casting it as a deep regression problem?
- What is the minimum amount of information that we need to provide such a system?  In particular, can we build a system that suggests—if it does not achieve—a viable solution that does not take an estimated position, as is computed in an astroinertial system or as is estimated by mariners through dead reckoning, as input?

*[Return to contents](#Contents)*


## <a id="System_Overview">2.0 System Overview


### 2.1 Use Case
We envision mariners using our system as a back-up to other forms of navigation, primarily for open-ocean navigation outside of five nautical miles from chartered hazards.  A navigator would depart on their voyage with a set of models suited for the vessel's track much in the same way as a navigator today loads relevant charts either from removable storage or the internet prior to departing on a voyage.  The system takes nothing more than the image returned from the edge device's installed camera and time from an accurate clock as input.  In its final implementation, it will return both raw position output and an industry standard NMEA output suitable for integration with an electronic charting package.  A navigator with internet access at sea can load additional models from the cloud as needed to account for adjustments to the planned voyage.


### 2.2 Assumptions
We made a number of engineering assumptions to make the problem tractable as a term project.

- We are simulating images that would be taken at sea with synthetic images.  We hope eventually to conduct an operational test of the system.  We assume, then, that the installed camera on the operational system can replicate the resolution and overall quality of our synthetic images.
- We assume the notional vessel on which the system would be installed is fitted with a three-axis gyrocompass to allow images to be captured at a fixed azimuth and elevation.  The requirements of the stabilization system can be inferred by perturbing the synthetic images to simulate stabilization error.


### 2.3 Components
Our system consists of a cloud component and an edge component.  An image generator creates batches of synthetic images, names them using a descriptive scheme that allows easy indexing by location and time, and stores the models in object storage buckets indexed by location and time.  The model trainer pulls from these buckets to create models specific to a bounded geographic area at a given with certain time bounds.  These models are stored in object storage.  The edge device—in this case a Jetson TX2—captures an image of the sky and the time at which the image was taken.  The inference engine performs a forward pass of the model, returning the vessel's predicted location as raw output.

![System diagram](https://github.com/gregtozzi/deep_learning_celnav/blob/main/report_images/system_diagram.png)

*[Return to contents](#Contents)*


## <a id="Model">3.0 The Model
Our model is a relatively simple CNN tuned to provide reasonably accurate predictions over a navigationally-relevant geographical area over a short but relevant time period.  We do not claim that this architecture is an ideal candidate for a production version of this project.  We merely want to demonstrate feasibility.

### 3.1 Model Architecture
Our goal was to explore architectures that could learn a vessel's position from an arbitrary image of the sky with the azimuth and elevation fixed and the time known.  We explored both DNNs and CNNs but settled on the latter because CNNs can encode usable predictions more efficiently relative to deep dense networks.

There are a number of ways to attack this problem.  Object detection could, for instance, be used to find the location of certain known celestial bodies in an image.  We chose to cast our research as a deep regression problem.  The literature suggests that convolutional neural networks have been applied to problems of head position detection [[9]](#9).  This is a similar problem to ours in the sense that we are trying to find a mapping between translations and rotations of elements of the sky and the position of the observer at a given time.  Our problem is different, however, in that the sky does not translate and rotate as a unitary whole.  The path of the moon and planets, for instance, is not fixed relative to that of the stars.  Stars do move together, however, on what is referred to simplistically as the _celestial sphere_ [[10]](#10).

Our network has two inputs.  The image input ingests pictures of the sky  the time input ingests the UTC time at which the image was taken.  The images are run through a series of convolutional and max pooling layers.  The results are concatenated with the normalized time and the resulting vector is put through dropout regularization, a dense hidden layer, and the regression head.  The head consists of two neurons, one each for normalized latitude and normalized longitude.  Latitude and longitude are normalized over the test area with the latitude of the southernmost bound mapping to 0 and the latitude of the northernmost bound mapping to 1.  Longitude is mapped similarly.  The output layer uses sigmoid activation to bound the output in the spatial domain on `([0,1], [0,1])`.

[![](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggVERcbiAgQVtJbWFnZSBJbnB1dF0gLS0-fE5vbmUsIDIyNCwgMjI0LCAxfCBCW0NvbnYyRCAtIDUgZmlsdGVycywgayA9IDEwXVxuICBCIC0tPnxOb25lLCAyMjQsIDIyNCwgMXwgQ1tGbGF0dGVuXVxuICBDIC0tPnxOb25lLCA1MDE3NnwgRFtDb25jYXRlbmF0ZV1cbiAgRCAtLT58Tm9uZSwgNTAxNzd8IEVbRGVuc2Ugdy9SZUx1IC0gMjU2XVxuICBFIC0tPnxOb25lLCAyNTZ8IEZbRHJvcG91dCAtIDAuMl1cbiAgRiAtLT58Tm9uZSwgMjU2fCBHW0RlbnNlIHcvU2lnbW9pZCAtIDJdXG4gIEhbVGltZSBJbnB1dF0gLS0-fE5vbmUsIHwgSVtGbGF0dGVuXVxuICBJIC0tPnxOb25lLCAxfCBEXG5cdFx0IiwibWVybWFpZCI6eyJ0aGVtZSI6Im5ldXRyYWwifSwidXBkYXRlRWRpdG9yIjpmYWxzZX0)](https://mermaid-js.github.io/mermaid-live-editor/#/edit/eyJjb2RlIjoiZ3JhcGggVERcbiAgQVtJbWFnZSBJbnB1dF0gLS0-fE5vbmUsIDIyNCwgMjI0LCAxfCBCW0NvbnYyRCAtIDUgZmlsdGVycywgayA9IDEwXVxuICBCIC0tPnxOb25lLCAyMjQsIDIyNCwgMXwgQ1tGbGF0dGVuXVxuICBDIC0tPnxOb25lLCA1MDE3NnwgRFtDb25jYXRlbmF0ZV1cbiAgRCAtLT58Tm9uZSwgNTAxNzd8IEVbRGVuc2Ugdy9SZUx1IC0gMjU2XVxuICBFIC0tPnxOb25lLCAyNTZ8IEZbRHJvcG91dCAtIDAuMl1cbiAgRiAtLT58Tm9uZSwgMjU2fCBHW0RlbnNlIHcvU2lnbW9pZCAtIDJdXG4gIEhbVGltZSBJbnB1dF0gLS0-fE5vbmUsIHwgSVtGbGF0dGVuXVxuICBJIC0tPnxOb25lLCAxfCBEXG5cdFx0IiwibWVybWFpZCI6eyJ0aGVtZSI6Im5ldXRyYWwifSwidXBkYXRlRWRpdG9yIjpmYWxzZX0)

### 3.2 Implementing a Haversine Loss Function
We want our loss function to minimize the navigational error returned by the model.  We initially used mean squared error across latitude and longitude as our loss function, but we found that convergence was slow and that the model frequently failed to converge to reasonable values.  There is, of course a nonlinear relationship between latitude and longitude.  Whereas a degree of latitude is consistently 60 nautical miles anywhere on the globe, lines of longitude converge at the poles and diverge at the equator.  Using mean squared error, then, results in inconsistent results in terms of the model's error in distance.  To correct this, we implemented a new loss function in TensorFlow based on the haversine formula which gives the great circle distance between two points defined by latitude and longitude [[11]](#11).

The haversine function is given below.  The value of interest, _d_, is the distance between two points defined by latitude longitude pairs [_φ_<sub>1</sub>, _λ_<sub>1</sub>] and [_φ_<sub>2</sub>, _λ_<sub>2</sub>].  The value _r_ is the radius of the Earth and sets the units.  We chose our _r_ to provide an output in nautical miles.

![System diagram](https://github.com/gregtozzi/deep_learning_celnav/blob/main/report_images/haversine.png)

The haversine function is strictly positive and is continuously differentiable in all areas of interest.  Minimizing the haversine loss minimizes the error between predicted and actual locations, and the negative gradient of the function gives the direction of steepest descent in terms of the predicted latitude and longitude.

*[Return to contents](#Contents)*

## <a id="Imgen">4.0 Generating Synthetic Images
We must rely on synthetic images to train the model because we cannot otherwise capture future configurations of the sky.  We adapted the open source astronomy program _Stellarium_ for use in the cloud as an image generator.  In this section we will detail our method for adapting Stellarium and will provide details for running the image generator in the cloud with images being stored in an S3 bucket.
 
 ### 4.1 Stellarium Overview
[Stellarium](https://stellarium.org) generates high-quality images of the sky for arbitrary geographic positions, times, azimuths (the heading of the virtual camera in true degrees), altitudes (the angular elevation of the virtual camera), camera field of view, and observer elevations [[12]](#12).  Stellarium's functionality is aimed at practitioners and hobbyists, it has been in continual development since 2006.  In addition to rendering celestial bodies, Stellarium can render satellites, meteor showers, and other astronomical events.  The program uses a JavaScript-based scripting language system to automate sequences of observations.

### 4.2 Containerizing Stellarium
Stellarium is designed for desktop operation.  Our Docker container allows the program to be run on a headless server in the cloud.  The basic container operation is outlined in the figure below.  A python script reads input from a YAML file and generates and outputs an SSC script that automates image generation.  Stellarium runs using a customized configuration file that prevents certain modules from running.  Stellarium executes the SSC script.  The image files are saved either to local storage on the host or to an S3 mount point that the container accesses on the host system.  From there, the images can be preprocessed using the Docker container discussed in section 5.1 below.

[![](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggVERcbiAgQVtZQU1MIEZpbGVdIC0tZnJvbSBob3N0LS0-IEIoU1NDIEdlbmVyYXRvciAtIFB5dGhvbilcbiAgQiAtLT4gQyhTdGVsbGFyaXVtKVxuICBDIC0tPiBEKFh2ZmIpXG4gIEQgLS1zY3JlZW4gY2FwdHVyZS0tPiBDXG4gIEMgLS10byBob3N0LS0-IEVbUzMgQnVja2V0XVxuICBDIC0tdG8gaG9zdC0tPiBGW0xvY2FsIFN0b3JhZ2VdXG4gIEYgLS1mcm9tIGhvc3QtLT4gRyhQcmVwcm9jZXNzb3IgLSBOdW1weSBvdXRwdXQpXG4gIEcgLS10byBob3N0LS0-IEVcbiAgRyAtLXRvIGhvc3QtLT4gRlxuXG5cdFx0IiwibWVybWFpZCI6eyJ0aGVtZSI6Im5ldXRyYWwifSwidXBkYXRlRWRpdG9yIjpmYWxzZX0)](https://mermaid-js.github.io/mermaid-live-editor/#/edit/eyJjb2RlIjoiZ3JhcGggVERcbiAgQVtZQU1MIEZpbGVdIC0tZnJvbSBob3N0LS0-IEIoU1NDIEdlbmVyYXRvciAtIFB5dGhvbilcbiAgQiAtLT4gQyhTdGVsbGFyaXVtKVxuICBDIC0tPiBEKFh2ZmIpXG4gIEQgLS1zY3JlZW4gY2FwdHVyZS0tPiBDXG4gIEMgLS10byBob3N0LS0-IEVbUzMgQnVja2V0XVxuICBDIC0tdG8gaG9zdC0tPiBGW0xvY2FsIFN0b3JhZ2VdXG4gIEYgLS1mcm9tIGhvc3QtLT4gRyhQcmVwcm9jZXNzb3IgLSBOdW1weSBvdXRwdXQpXG4gIEcgLS10byBob3N0LS0-IEVcbiAgRyAtLXRvIGhvc3QtLT4gRlxuXG5cdFx0IiwibWVybWFpZCI6eyJ0aGVtZSI6Im5ldXRyYWwifSwidXBkYXRlRWRpdG9yIjpmYWxzZX0)

[_Installing and running the image generator_](https://github.com/gregtozzi/deep_learning_celnav/tree/master/image_generator)

*[Return to contents](#Contents)*

## <a id="Train">5.0 Training the Model

We built a Docker container to facilitate training in the cloud.  The container is built on the base TensorFlow container [[13]](#13) and facilitates deploying instances to allow simultaneous training of models representing different spatial-temporal regions.  Our model is written in TensorFlow and requires two inputs—one image and one floating point value identifying the time at which the image was generated.

### 5.1 Preprocessing
We provide a Docker container to handle the task of preprocessing images into Numpy arrays ready to be used in training.  The container converts images into Numpy arrays with dimensions `(None, 224, 224, 1)`  with a script that leans heavily on [OpenCV](https://opencv.org) [[14]](#14).  Images are read from a single  directory.  The images are reduced to the the target resolution and are stacked in a single Numpy array.  Time and true position are parsed from the input file names which follow the format `<lat>+<long>+<YYYY>-<MM>-<DD>T<hh>:<mm>:<ss>.png`.  Date time groups are converted to `numpy.datetime64` and are normalized on `[0,1]` based on the temporal bounds on the training set.  Latitudes and longitudes are normalized on `[0,1]` based on the maximum extent of the geographic area under consideration and formed into an array with dimensions `(None, 2)`.

The data are broken into training and validation sets based on a user-provided split and random assignment.

[_Installing and running the preprocessor_](https://github.com/gregtozzi/deep_learning_celnav/tree/master/training/preprocessor)

### 5.2 Training
We use the [ktrain](https://towardsdatascience.com/ktrain-a-lightweight-wrapper-for-keras-to-help-train-neural-networks-82851ba889c) package to train the models [[15]](#15).  The `tensorflow.keras.model` object is wrapped in a `ktrain.learner` object to simplify training and access ktrain's built in callbacks.  We train using the `ktrain.learner.autofit` function with an initial maximum learning rate of 2-4.  The training function applies the triangular learning rate policy with reduction in maximum learning rate when a plateau is encountered on the validation loss introduced by Smith [[16]](#16).

The container saves models in the `.h5` format to a user-specified directory that maps to the host.  In this way, the model can be saved either locally or to object storage.

We established an experimental space from 36°N to 40°N, from 074°W to 078°W, and from 2020-05-25T22:00:00 UTC to 2020-05-26T02:00:00 UTC.  We trained the model using 6,788 images reduced to single channel 224 x 224.  We validated on 715 images similarly reduced in color and resolution.  The base images were of different aspect ratios.  The training routine uses a batch size of 32.  The figure below details the application of the triangular learning rate policy.

![System diagram](https://github.com/gregtozzi/deep_learning_celnav/blob/main/report_images/lr_policy.png)

Training loss converged to approximately 5.5 nautical miles and validation loss converged to just over 6.5 nautical miles after 52 epochs as shown below.

![System diagram](https://github.com/gregtozzi/deep_learning_celnav/blob/main/report_images/train_val_loss.png?raw=true)

[_Installing and running the training container_](https://github.com/gregtozzi/deep_learning_celnav/tree/master/training)

### 5.3 Performance on a Notional Vessel Transit
We consider a four-hour period in the test area defined above.  A notional vessel is on a course of 208° true—that is, referenced to true north—at a speed of 12 nautical miles per hour or _knots_.  A nautical mile is slightly longer than a statute mile.  The vessel employs our system to fix its position hourly beginning at 22:00 UTC.  The chart below is overlaid with the vessel's actual positions in blue and predicted positions in red.

![Notional transit](https://github.com/gregtozzi/deep_learning_celnav/blob/main/report_images/chart_12200.png?raw=true)

Our model is trained using a spatial-temporal grid based on our intuition that the model is learning an interpolation and should, therefore, be trained on data spaced at regular intervals.  Spatial interpolation was generally quite good.  Temporal interpolation was more challenging, likely owing to our decision to limit the temporal density to 20-minute intervals during training.  Mariners typically take fixes at regular intervals on the hour, half hour, quarter hour, etc..., and our fix interval for the example above aligns with times trained on the temporal grid.  We leave improving temporal interpolation as an area for further study.

*[Return to contents](#Contents)*

## <a id="Edge">6.0 Inferencing at the Edge

### 6.1 The Camera Container and Broker

The edge implementation consists of three Docker containers on a Jetson connected via the MQTT protocol.  The camera container 'creates' an image (in this case synthetically) and forwards it via the MQTT broker to the TensorFlow-powered inference container.  Since we are using synthetic images, our current implementation of the camera container takes an image from a directory of images and forwards it for inferencing.  The camera container displays images as they are loaded to assist in troubleshooting.

The camera container does much of the image preprocessing (primarily reducing any image to 224x224 and converting it to black and white).  We do this at the level of the camera to reduce the size of the messages being passed through the MQTT protocol, thus making the message network more reliable.

The MQTT broker sits between the camera container and the inference container and acts as a bridge to pass images (and the file name which has the ground truth embedded it and is used for assessing prediction accuracy).

[_Installing and running the camera and broker containers_](https://github.com/gregtozzi/deep_learning_celnav/tree/master/inference/edge_network)

### 6.2 Inference Container

The inference container runs the TensorFlow model that was trained in the cloud.  We found transferring models using the older `.h5` format to be more reliable than more current Keras methods.  On receipt of an image, the container further preprocesses the image, feeds the processed image forward through the network and displays the output as text.  We also provide a measure of accuracy (using the ground truth which is embedded in the file names passed through).

The screenshot below shows the edge device at work.  In the lower right you have the camera container.  In the lower left you have the test image it generated (after preprocessing).  In the upper left is the broker that provides communication to the inference engine.  And in the upper right is the inference container with readouts from two different prediction requests sent to it by the camera container.

![Camera, broker and inference engine at work](https://github.com/gregtozzi/deep_learning_celnav/blob/master/report_images/screenshot_of_inference.png)

[_Installing and running the inference container_](https://github.com/gregtozzi/deep_learning_celnav/tree/master/inference)

*[Return to contents](#Contents)*

## <a id="End">7.0 An End-to-End Example

This section walks through a detailed example of generating synthetic data, training a model in the cloud, pushing that model to an object store, pulling the model to the edge device, capturing an image locally, and making an inference based on that image.

### 7.1 Generate Images
We begin by provisioning a virtual server on a cloud provider.  Rather to storing to object storage, we will store the images locally to train the model on the same virtual server.  We create a local directory to hold the images.  We then clone the repo and `cd` into the `image_generator` directory.

```
git clone https://github.com/gregtozzi/deep_learning_celnav.git
mkdir /data/image_train_val
mkdir /data/image_test
cd w251-project/image_generator
```

Build the docker image.

```
docker build -t imgen -f image_generator.dockerfile .
```

For this case, we will use the `ssc_gen.yml` file as it appears in the repository.  We want to build a total of 20,000 images.  To speed the process we will use 10 workers.

```
for i in 0 1 2 3 4 5 6 7 8 9
do docker run --name imgen$i -dit -v /data/image_train_val:/data/image_train_val imgen
docker cp ssc_gen.yml imgen$i:/
docker exec -d imgen$i ./screenshot.sh
done
```

Get rid of the now-unneeded workers.

```
for i in 0 1 2 3 4 5 6 7 8 9
do docker stop imgen$i
docker rm imgen$i
done
```

Edit the `ssc_gen.yml` file to change the target directory to `/data/image_test` and build 1,000 test images.

### 7.2 Preprocessing the Images
We will preprocess the image files in place in preparation for training.  To do this, `cd` to the preprocessor directory and build the dockerfile.

```
cd /root/w251-project/training/preprocessor
docker build -t preprocessor -f preprocessor.dockerfile .
```

We will make a new directory to store our processed Numpy arrays.

```
mkdir /data/sets
```

Edit the `preprocessor.yml` file as necessary, spin up a container, and preprocess the images.

```
docker run --name preproc -dit -v /data/image_train_val:/data/image_train_val -v /data/image_test:/data/image_test -v /data/sets:/data/sets preprocessor python3 preprocessor.py
```

Once the preprocessing is complete and you have verified that the `.npy` files are where you expect, tear down the container.

```
docker rm preproc
```

### 7.3 Training the Model

Build the docker image.

```
cd /root/w251-project/training
docker build -t training -f training.dockerfile .
```

Create a directory to house models.

```
mkdir /data/models
```

Spin up the container and train a model.

```
nvidia-docker run -dit --name train -v /data/sets:/data/sets -v/data/models:/data/models tensorflow/tensorflow:latest-gpu-jupyter python3 /training.py
```

You should now see your trained model in `data/models` on the host machine.

### 7.4 Spooling up Edge Device for Images and Predictions

[_Installing and running the camera and broker containers_](https://github.com/gregtozzi/deep_learning_celnav/tree/master/inference/edge_network)

[_Installing and running the inference container_](https://github.com/gregtozzi/deep_learning_celnav/tree/master/inference)



*[Return to contents](#Contents)*

## References

<a id="1">[1]</a> J. Kasper and C. Hutchinson, "The Omega navigation system--An overview," in  _IEEE Communications Society Magazine_, vol. 16, no. 3, pp. 23-35, May 1978, doi: 10.1109/MCOM.1978.1089729.

<a id="2">[2]</a> W. J. Thrall, "Loran-C, an Overview," in _Proc. 13th Annu. Precise Time Planning Meet_., p. 449, 1976. and _Time Interval Applications and Planning Meet_., NASA Conf. Publ. 2220, p. 121, 1981

<a id="3">[3]</a> G. Beutler, W. Gurtner, M. Rothacher, U. Wild and E. Frei, "Relative Static Positioning with the Global Positioning System:  Basic Technical Considerations," in _Global Positioning System:  An Overview_., International Association of Geodesy Symposia 102, ch. 1, 1990.

<a id="4">[4]</a> E. G. Blackwell, "Overview of Differential GPS Methods" in _Navigation_, 32: 114-125, 1985. doi:[10.1002/j.2161-4296.1985.tb00895.x](https://doi.org/10.1002/j.2161-4296.1985.tb00895.x)

<a id="5">[5]</a> U. S. Coast Guard Navigation Center, "Special Notice Regarding LORAN Closure", [https://www.navcen.uscg.gov/?pageName=loranMain](https://www.navcen.uscg.gov/?pageName=loranMain).

<a id="6">[6]</a> T. Hitchens, "SASC Wants Alternative GPS By 2023," 29 June 2020, [breakingdefense.com/2020/06/sasc-wants-alternative-gps-by-2023/](breakingdefense.com/2020/06/sasc-wants-alternative-gps-by-2023/.).

<a id="7">[7]</a> M. Garvin, "Future of Celestial Navigation and the Ocean-Going Military Navigator," [OTS Master's Level Projects & Papers. 41](https://digitalcommons.odu.edu/ots_masters_projects/41), 2010.

<a id="8">[8]</a> R. Whitman, "Astroinertial Navigation for Cruise Applications." Northrop Corporation, 1980.

<a id="9">[9]</a> S. Lathuiliere, P. Mesejo, X. Alameda-Pineda and R. Horaud, "A Comprehensive Analysis of Deep Regression," [arXiv:1803.08450v2](https://arxiv.org/pdf/1803.08450.pdf), 2019.

<a id="10">[10]</a> NATIONAL GEOSPATIAL-INTELLIGENCE AGENCY, _Pub No. 9, American Practical Navigator: an Epitome of Navigation_, 2017.

<a id="11">[11]</a> C. N. Alam, K. Manaf, A. R. Atmadja and D. K. Aurum, "Implementation of haversine formula for counting event visitor in the radius based on Android application," _2016 4th International Conference on Cyber and IT Service Management_, Bandung, 2016, pp. 1-6, doi: [10.1109/CITSM.2016.7577575]([https://ieeexplore.ieee.org/abstract/document/7577575](https://ieeexplore.ieee.org/abstract/document/7577575)).

<a id="12">[12]</a> M. Gates, G. Zotti and A. Wolf, _Stellarium User Guide_, 2016. doi:[10.13140/RG.2.2.32203.59688](https://www.researchgate.net/publication/306257191_Stellarium_User_Guide)

<a id="13">[13]</a> “Docker | TensorFlow” _TensorFlow_, TensorFlow.org, 15 July 2020, https://www.tensorflow.org/install/docker.

<a id="14">[14]</a> G. Bradski, G., "The OpenCV Library" _Dr. Dobb&#39;s Journal of Software Tools_, 2000.

<a id="15">[15]</a> A. Maiya, "ktrain: A Low-Code Library for Augmented Machine Learning," [arXiv:2004.10703](https://arxiv.org/abs/2004.10703), 2020.

<a id="16">[16]</a> L. Smith, "Cyclical Learning Rates for Training Neural Networks," [arXiv:1506.01186v6](https://arxiv.org/abs/1506.01186v6), 2017.

*[Return to contents](#Contents)*

