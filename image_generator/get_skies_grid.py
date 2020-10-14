import yaml
import subprocess
from get_skies_helper import get_file_name, timelinspace
import random
import numpy as np
    

def main():
    """
    Parses a YAML file containing lists of positions from which to
    collect images.  Generates a ssc script to automate the process
    in Stellarium.  Runs the ssc script in Stellarium.
    """
    
    # Parse the yaml file
    with open('ssc_gen.yml') as config_file:
        # The default version of Pyyaml with the NVIDIA image does
        # not include FullLoader, so we have to fall back to the
        # pre-2019 syntax
        config_data = yaml.load(config_file)
    
    image_dir     = config_data['image_path']
    azi           = config_data['azi']
    alt           = config_data['alt']
    fov           = config_data['fov']
    latstart      = config_data['latstart']
    latend        = config_data['latend']
    latsteps      = config_data['latsteps']
    longstart     = config_data['longstart']
    longend       = config_data['longend']
    longsteps     = config_data['longsteps']
    dtstart       = config_data['dtstart']
    dtend         = config_data['dtend']
    dtsteps       = config_data['dtsteps']
    
    # Calculate the steps that need to be taken in lat and long
    lats  = np.linspace(latstart, latend, latsteps)
    longs = np.linspace(longstart, longend, longsteps)
    times = timelinspace(dtstart, dtend, dtsteps)

    # Generate an ssc file that we can feed to Stellarium
    image_index = 0
    
    with open('/usr/share/stellarium/scripts/get_multi_sky.ssc', 'w') as f:
        for lat in lats:
            for long in longs:
                for time in times:
                    print('Generating file for {}, {}, {}'.format(lat, long, time))
                    
                    # Write the script
                    f.write('core.setObserverLocation({}, {}, 15, 0, "Ocean", "Earth");\n'.format(lat, long))
                    f.write('core.setDate("{}", "utc");\n'.format(time))
                    f.write('core.moveToAltAzi({}, {});\n'.format(alt, azi))
                    f.write('LandscapeMgr.setFlagAtmosphere(true);\n')
                    f.write('StelMovementMgr.zoomTo({},0);\n'.format(fov))
                    f.write('core.setGuiVisible(false);\n')
                    f.write('StarMgr.setLabelsAmount(0);\n')
                    f.write('SolarSystem.setFlagLabels(false);\n')
                    f.write('MeteorShowers.setEnableMarker(false);\n')
                    f.write('LandscapeMgr.setFlagLandscapeSetsLocation(false);\n')
                    f.write('LandscapeMgr.setCurrentLandscapeName("Ocean");\n')
                    f.write('core.wait(1);\n')
                    
                    # Settings to save the screenshot
                    file_name = get_file_name(lat, long, time)
                    f.write('core.screenshot("{}", invert=false, dir="{}", overwrite=true);\n'.format(file_name, image_dir))

    # Open Stellarium and run the script
    #proc_stellarium = subprocess.Popen(['/Applications/Stellarium.app/Contents/MacOS/stellarium',
    #                                    '--startup-script', 'get_multi_sky.ssc', '--screenshot-dir',
    #                                    image_dir], stdout=subprocess.PIPE) 

if __name__ == "__main__":
    main()


