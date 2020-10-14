import yaml
import subprocess
import random
import numpy as np

def get_file_name(lat,long,dt):
    file_name=str(lat)+"+"+str(long)+"+"+dt
    #print (file_name)
    return file_name

def generate_dt(dtstart, dtend):
    """
    Given a starting time and an ending time,
    generate a grid of equally-spaced intermediate
    times.
    
    Args:
        dtstart: str - Must be able to be converted
               to an np.datetime64 object
        dtend:  str - Must be able to be converted
               to an np.datetime64 object

               
    Returns:   a dt converted to strings
    """
    # Convert to datetime64 to make calculations
    # easier
    start_dt = np.datetime64(dtstart)
    end_dt   = np.datetime64(dtend)
    
    time_diff = end_dt - start_dt
    print('time diff',time_diff)
    random_time_diff=random.uniform(0,time_diff)
    print('random_time_diff',random_time_diff)
    time = start_dt +random_time_diff    
    print('time',time)
    
    return str(time)


def main():
    """
    Parses a YAML file containing lists of positions from which to
    collect images.  Generates a ssc script to automate the process
    in Stellarium.  Runs the ssc script in Stellarium.
    """

    # Parse the yaml file
    with open('test_images.yml') as config_file:
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

    # Write the ssc file
    image_index = 0
    with open(script_path + 'get_multi_sky.ssc', 'w') as f:
        # TODO:  Allow time to be read in as a list (move the following
        #        line to the loop).

        for x in range(number_images):
            #pick random location within grid
            i=random.uniform(latstart,latend)
            j=random.uniform(longstart,longend)
            #pick random dt within range
            dt=generate_dt(dtstart,dtend)
            print (x,i,j)
            #write script for gathering image from that location
            f.write('core.setObserverLocation({}, {}, 15, 0, "Ocean", "Earth");\n'.format(i, j))
            f.write('core.setDate("{}", spec="local");\n'.format(dt))
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

            # The next statement needs to be repeated to generate stable images.
            # It seems that Stellarium doesn't like altitudes of 90 degrees.
            #file_name=get_file_name(i,j,dt)
            f.write('core.screenshot("{}", invert=false, dir="{}", overwrite=true);\n'.format(get_file_name(i,j,dt), image_dir))
            
    # Open Stellarium and run the script
    proc_stellarium = subprocess.Popen(['stellarium', '--startup-script', 'get_multi_sky.ssc', '--screenshot-dir', image_dir], stdout=subprocess.PIPE)

if __name__ == "__main__":
    main()


