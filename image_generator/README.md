[_Return to the white paper_](https://github.com/gregtozzi/deep_learning_celnav#Train)

# Synthetic Image Generation in the Cloud
## W251 - Final Project - Tozzi/Metz - Summer 2020
### Setting up Containers on a Virtual Server for Image Generation

#### Files
The `image_generator` directory contains the following files:

`prepare_vs.sh` prepares the cloud server to save image files to an object store and builds the Dockerfile.

`image_generator.dockerfile` builds the `imgen` image.

`get_skies.py` reads a YAML input file and generates an SSC that Stellarium executes.  The SSC is build from random locations sampled from the spatial-temporal grid defined in the YAML file.

`get_skies.sh` and `get_skies_grid.py` are deprecated but is kept for reference.  This files contains code to generate images based on a spatial-temporal grid.

`get_skies_helper.py` contains functions used by `get_skies.py`.

`screenshot.sh` runs the image generation routine from the docker container.

`default_cfg.ini` is a replacement for Stellarium's configuration file.  This turns off several modules that slow the image generation process.

`ssc_gen.yml` is a sample input file.

#### Installing and Running the Image Generator
>Note:  The image generator must be run on a server with a GPU.  We assume that the virtual server is running Ubuntu 18.04 and is already provisioned with Docker and s3fs.

>Note:  The output of the image generator need not be placed in an object store.  The steps below can be adapted to direct the images to local storage.

- After pulling the repository on the virtual server, create a credentials file with `vi credentials` to allow access to object storage.  The credentials file should contain a single line with `<api_key>:<secret>`.

- Next edit the `prepare_vs.sh` script such that it contains the user's object store information, and run script to build the image generator Docker image and link the mountpoint to the appropriate S3 bucket.

```
chmod +x prepare_vs.sh
./prepare_vs.sh
```

- Verify that the `imgen` image is built by calling `docker images`.

- Edit the `ssc_gen.yml` file to configure the maximum and minimum latitudes, longitudes, and times that will be rendered.

- Create an `imgen` container instance with access to the host's mountpoint. `docker run --name <name> -dit -v /<mountpoint>:/<mountpoint> imgen`

- Copy the YAML file to the container.  `docker cp ssc_gen.yml <name>:/`

- Run the image generator and monitor progress.  `docker exec <name> ./screenshot.sh`

The steps above can be automated further as needed.  Multiple instances of the image generator container can be run on the same server to speed the process.
