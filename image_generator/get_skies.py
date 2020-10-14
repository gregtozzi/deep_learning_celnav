import yaml
from get_skies_helper import get_file_name
import numpy as np
import random

def main():
    """
    Parses a YAML file containing lists of positions from which to
    collect images.  Generates a ssc script to automate the process
    in Stellarium.
    """
    # Parse the yaml file
    with open('ssc_gen.yml') as config_file:
        # The default version of Pyyaml with the NVIDIA image does
        # not include FullLoader, so we have to fall back to the
        # pre-2019 syntax
        config_data = yaml.load(config_file)
    
    #TODO:  This can be done better with an eval function and a loop
    image_dir     = config_data['image_path']
    azi           = config_data['azi']
    alt           = config_data['alt']
    fov           = config_data['fov']
    latstart      = config_data['latstart']
    latend        = config_data['latend']
    longstart     = config_data['longstart']
    longend       = config_data['longend']
    dtstart       = config_data['dtstart']
    dtend         = config_data['dtend']
    number_images = config_data['number_images']
    
    start_dt = np.datetime64(dtstart)
    end_dt   = np.datetime64(dtend)
    delta_t  = end_dt - start_dt

    # Write the ssc file
    image_index = 0
    with open('/usr/share/stellarium/scripts/get_multi_sky.ssc', 'w') as f:

        for x in range(number_images):
            #pick random location within grid
            i = random.uniform(latstart,latend)
            j = random.uniform(longstart,longend)
            t = str(random.uniform(0, 1) * delta_t + start_dt)
            print (x,i,j)
            #write script for gathering image from that location
            f.write('core.setObserverLocation({}, {}, 15, 0, "Ocean", "Earth");\n'.format(i, j))
            f.write('core.setDate("{}", "utc");\n'.format(t))
            f.write('core.moveToAltAzi({}, {});\n'.format(alt, azi))
            f.write('LandscapeMgr.setFlagAtmosphere(true);\n')
            f.write('StelMovementMgr.zoomTo({},0);\n'.format(fov))
            f.write('core.setGuiVisible(false);\n')
            f.write('StarMgr.setLabelsAmount(0);\n')
            f.write('SolarSystem.setFlagLabels(false);\n')
            f.write('LandscapeMgr.setFlagLandscapeSetsLocation(false);\n')
            f.write('LandscapeMgr.setCurrentLandscapeName("Ocean");\n')
            f.write('core.wait(1);\n')            

            file_name=get_file_name(i,j,t)
            f.write('core.screenshot("{}", invert=false, dir="{}", overwrite=true);\n'.format(file_name, image_dir))
            

if __name__ == "__main__":
    main()


