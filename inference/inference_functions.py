#inference_functions.py
import cv2
import numpy as np

def scale_down(numbers,top,bottom):
    top=max(numbers)
    bottom=min(numbers)
    number_range=top-bottom
    revised=[x-bottom for x in numbers]
    revised=[x/number_range for x in revised]
    return revised,top,bottom

def scale_up(number,top,bottom):
    number_range=top-bottom
    revised=number*number_range 
    revised=revised+bottom
    return revised

def load_image(image_path, dim=(224,224), channels=1):
    """
    Loads a single image as a Numpy array and resizes it as
    desired.  The default dimensions are consistent with
    those expected by the VGG models.  
    Args:
    image_path: str pointing to the file
    dim: Two-element tuple giving the desired height
         and width of the processed image
    Returns:
    image:  A single-channel Numpy array
    """
    if channels == 1:
        image = cv2.imread(image_path, 0)
    else:
        image = cv2.imread(image_path, 1)
    image = cv2.resize(image, dim)
    #, interpolation = cv2.INTER_AREA
    return image
    
    
def build_input(image_dir, channels=1, dim=(224,224)):
    """
    Loads all of the images into a single numpy array.
    Assumes that there are 101 equally-spaced images
    spanning lattitudes from 35N to 45N.  
    Args:
        image_dir: str giving name of the image directory
    Returns:
    X:  A 3-dimensional numpy array containing the
        images. Image height and width are set by
        `load_images` and default to 224 x 224.
    y:  A 1-dimensional numpy array of target lattitudes.
    """
    X = []
    files = os.listdir(image_dir)
    for file in files:
        if file[-4:] == '.png':
            image_path = os.path.join(image_dir, file)
            image = load_image(image_path, channels=channels, dim=dim)
            X.append(image)
    return (np.array(X) / 255)


def get_labels(file):
    """
    Parses file names to extract lat, long, and time.
    """
    file_split = file.split('+')
    lat = float(file_split[0])
    long = float(file_split[1])
    time = file_split[2].split('.')[0]
    #dtype='datetime64'
    return lat,long,time


def normalize_times(time,dtstart,dtend):
    """
    Converts times to a float bounded by [0,1]
    
    Args:
        times: numpy.array with dtype datetime64
        
    Returns:
        numpy.array of decimal times bounded on [0,1]
    """
    time=np.datetime64(time)
    dtstart=np.datetime64(dtstart)
    dtend=np.datetime64(dtend)
    time_range = (dtend - dtstart).astype('float64')
    seconds_from_t0 = (time - dtstart).astype('float64')
    return seconds_from_t0 / time_range

    
    
def haversine_loss(y_true, y_pred, denorm=(36.0, 4.0, -78.0, 4.0), R=3443.92):
    """
    Returns the mean squared haversine distance
    between arrays consisting of lattitudes and
    longitudes.
    
    Args:
        y_true:  Either an np.array or a tf.constant
                 of dimensions m x 2 where m in the
                 number of observations.  Each row is
                 an ordered pair of [lat, long].
                 
        y_pred:  Has the same form as y_true.
        
        dnorm:   A tuple of four values needed to
                 convert normalized lat and long back
                 to actual values.
        
        R:       Float giving the radius of the earth.
                 The default value is in nautical
                 miles.  Values in other units:
                 
                 kilometers    -> 6378.14
                 statute miles -> 3963.19
                 smoots        -> 3.748e+6
        
    Returns:
        tf.tensor of shape () and dtype float64 giving
        the mean square distance error using the
        haversine function.
    
    Examples:
    
        Input:
        y1     = np.array([[0, 0]])
        y_hat1 = np.array([[0, 180]])
        
        Expected result:
        (pi * R) ** 2 = 117059281.6 nm^2
        
        Input:
        y2     = np.array([[0, 0]])
        y_hat2 = np.array([[90, 0]])
        
        Expected result:
        (pi * R / 2) ** 2 = 29264820.4 nm^2
        
        Input:
        Portmsouth, VA to Rota, Spain
        y3     = tf.constant([[36.8354, -76.2983]])
        y_hat3 = tf.constant([[36.6237, -6.3601]])
        
        Expected result:
        37065212.0 km^2
        
    Notes:
        Closely follows the JS implmentation at
        https://www.movable-type.co.uk/scripts/latlong.html.
    """
    # Break inputs into lattitudes and longitudes for
    # convienience

    # Convert normalized lat and long into actuals
    lat_min, lat_range, long_min, long_range = denorm
    lat1  = y_true[:,0] * lat_range + lat_min
    lat2  = y_pred[:,0] * lat_range + lat_min
    long1 = y_true[:,1] * long_range + long_min
    long2 = y_pred[:,1] * long_range + long_min
    
    # Compute phis and lambdas 
    phi1 = lat1 * np.pi / 180
    phi2 = lat2 * np.pi / 180
    delta_phi    = (lat2 - lat1) * np.pi / 180
    delta_lambda = (long2 - long1) * np.pi / 180
    
    # Intermediate computations
    a = tf.square(tf.sin(delta_phi / 2)) + tf.cos(phi1) * tf.cos(phi2) * tf.square(tf.sin(delta_lambda / 2))
    c = 2 * tf.atan2(tf.sqrt(a), tf.sqrt(1 - a))
    
    # Compute distances
    d = R * c
    
    # Compute the mean squared distance (MSE)
    return tf.reduce_mean(d)