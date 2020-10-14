#!/bin/sh

# Generate the .ssc file
python3 get_skies.py

# Set up xvfb to run Stellarium headless
Xvfb :89 -ac -screen 0 1024x768x24+32 &
export DISPLAY=":89"
export LD_PRELOAD=/usr/lib/fglrx/libGL.so.1.2.xlibmesa

# Generate the files and save them to the cloud
stellarium --startup-script get_multi_sky.ssc
killall Xvfb