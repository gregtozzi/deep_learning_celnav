import yaml
import subprocess
#import geopy
#from sky_helper import next_lat, next_long,get_file_name
from sky_helper import get_file_name
import random
import numpy as np

def main():
    """
    Parses a YAML file containing lists of positions from which to
    collect images.  Generates a ssc script to automate the process
    in Stellarium.  Runs the ssc script in Stellarium.
    
    TODO: Save resulting images to a cloud object store.
    """
    # Parse the yaml file
    with open('ssc_generator.yml') as config_file:
        # The default version of Pyyaml with the NVIDIA image does
        # not include FullLoader, so we have to fall back to the
        # pre-2019 syntax
        config_data = yaml.load(config_file)
    
    script_path = config_data['script_path']
    image_dir = config_data['image_path']
    azi = config_data['azi']
    alt = config_data['alt']
    fov = config_data['fov']
    latstart = config_data['latstart']
    latend=config_data['latend']
    longstart = config_data['longstart']
    longend=config_data['longend']
    dtstart = config_data['dtstart']
    dtend=config_data['dtend']
    number_images=config_data['number_images']

    start_dt=np.datetime64(dtstart)
    end_dt=np.datetime64(dtend)
    delta_t=end_dt-start_dt

    # Write the ssc file
    image_index = 0
    with open(script_path + 'get_multi_sky.ssc', 'w') as f:
        # TODO:  Allow time to be read in as a list (move the following
        #        line to the loop).

        for x in range(number_images):
            #pick random location within grid
            i=random.uniform(latstart,latend)
            j=random.uniform(longstart,longend)
            t=str(random.uniform(0,1)*delta_t+start_dt)
            print (x,i,j)
            #write script for gathering image from that location
            f.write('core.setObserverLocation({}, {}, 15, 0, "Ocean", "Earth");\n'.format(i, j))
            f.write('core.setDate("{}", spec="local");\n'.format(t))
            f.write('core.moveToAltAzi({}, {});\n'.format(alt, azi))
            f.write('LandscapeMgr.setFlagAtmosphere(true);\n')
            f.write('StelMovementMgr.zoomTo({},0);\n'.format(fov))
            f.write('LandscapeMgr.setFlagCardinalsPoints(false);\n')
            f.write('core.setGuiVisible(false);\n')
            f.write('StarMgr.setLabelsAmount(0);\n')
            f.write('SolarSystem.setFlagLabels(false);\n')
            f.write('MeteorShowers.setEnableMarker(false);\n')
            f.write('LandscapeMgr.setFlagLandscapeSetsLocation(false);\n')
            f.write('LandscapeMgr.setCurrentLandscapeName("Ocean");\n')
            f.write('core.wait(1);\n')

            file_name=get_file_name(i,j,t)
            f.write('core.screenshot("{}", invert=false, dir="{}", overwrite=true);\n'.format(get_file_name(i,j,t), image_dir))
            
    # Open Stellarium and run the script
    proc_stellarium = subprocess.Popen(['stellarium', '--startup-script', 'get_multi_sky.ssc', '--screenshot-dir', image_dir], stdout=subprocess.PIPE)

if __name__ == "__main__":
    main()


