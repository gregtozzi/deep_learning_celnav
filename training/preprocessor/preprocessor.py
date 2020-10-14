import cv2
import os
import numpy as np
import yaml


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
    image = cv2.resize(image, dim)#, interpolation = cv2.INTER_AREA)
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


def build_labels(image_dir):
    """
    Parses file names to extract lat, long, and time.
    
    Args:
        image_dir: str giving name of the image directory
        
    Returns:
        numpy.array of lats and longs with dimensions
        len(image_dir) x 2
        
        numpy.array of times
    """
    files = os.listdir(image_dir)
    y = []
    times = []
    for file in files:
        if file[-4:] == '.png':
            file_split = file.split('+')
            lat = float(file_split[0])
            long = float(file_split[1])
            time = file_split[2].split('.')[0]
            y.append((lat, long))
            times.append(time)
    return np.array(y), np.array(times, dtype='datetime64')


def normalize_times(times, t_min, t_max):
    """
    Converts times to a float bounded by [0,1]
    
    Args:
        times: numpy.array with dtype datetime64
				t_min: time to fix as 0
				t_max: time to fix as 1
        
    Returns:
        numpy.array of decimal times bounded on [0,1]
    """
    t_min = np.datetime64(t_min)
    t_max = np.datetime64(t_max)
    time_range = (t_min - t_max).astype('float64')
    seconds_from_t0 = (times - t_min).astype('float64')
    
    return seconds_from_t0 / time_range


def normalize_y(y, lat_min, lat_range, long_min, long_range):
    """
    Converts lats and longs to values bounded by [0,1]
    
    Args:
        y:      numpy.array of dimensions observations x 2
				lat_min:    lat to set as 0
				lat_range:  difference between min and max lat in degrees
				long_min:   long to set as 0
				long_range: difference between min and max long in degrees
        
    Returns
        numpy.array: normalized values
        
        tuple: values needed by the Haversine loss function
    """  
    y_norm = np.zeros(y.shape)
    
    y_norm[:,0] = (y[:,0] - lat_min) / lat_range
    y_norm[:,1] = (y[:,1] - long_min) / long_range
    
    return y_norm, (lat_min, lat_range, long_min, long_range)

def main():
    # Parse the yaml file
    with open('preprocessor.yml') as config_file:
        config_data = yaml.load(config_file)
        train_val_path = config_data['train_val_path']
        test_path      = config_data['test_path']
        array_path     = config_data['array_path']
        latstart       = config_data['latstart']
        latrange       = config_data['latrange']
        longstart      = config_data['longstart']
        longrange      = config_data['longrange']
        dtstart        = config_data['dtstart']
        dtend          = config_data['dtend']
        tv_split       = config_data['tv_split']
    
    # Read in the training and validation data
    print('Loading training and validation sets')
    X = build_input(train_val_path, channels=1, dim=(224,224))
    X = np.expand_dims(X, 3)
    y, times = build_labels(train_val_path)
    y_norm, denorm = normalize_y(y, latstart, latrange, longstart, longrange)
    
    # Split into training and validation sets
    print('Splitting training and validation sets')
    random_draw = np.random.uniform(low=0, high=1, size=len(X))
    test_mask = random_draw <= tv_split
    norm_times = normalize_times(times, dtstart, dtend)

    x_val = X[test_mask]
    y_val = y_norm[test_mask]
    t_val = norm_times[test_mask]

    x_train = X[~test_mask]
    y_train = y_norm[~test_mask]
    t_train = norm_times[~test_mask]
    
    print('Saving training and validations npy files')
    np.save(os.path.join(array_path, 'x_train.npy'), x_train)
    np.save(os.path.join(array_path, 'x_val.npy'), x_val)
    np.save(os.path.join(array_path, 'y_train.npy'), y_train)
    np.save(os.path.join(array_path, 'y_val.npy'), y_val)
    np.save(os.path.join(array_path, 't_train.npy'), t_train)
    np.save(os.path.join(array_path, 't_val.npy'), t_val)
    
    # Read in the test set
    print('Loading test set')
    x_test = build_input(train_val_path, channels=1, dim=(224,224))
    y, times = build_labels(train_val_path)
    y_test, denorm = normalize_y(y, latstart, latrange, longstart, longrange)
    t_test = normalize_times(times, dtstart, dtend)
    
    print('Saving test npy files')
    np.save(os.path.join(array_path, 'x_test.npy'), x_test)
    np.save(os.path.join(array_path, 'y_test.npy'), y_test)
    np.save(os.path.join(array_path, 't_test.npy'), t_test)
    
    
if __name__ == "__main__":
    main()
    
