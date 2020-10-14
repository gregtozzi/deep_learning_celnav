import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Lambda, Concatenate, Flatten, concatenate
from tensorflow.keras import Model
import ktrain
import random
import yaml
import os


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
    
    
def main():
    # Parse the yaml file
    with open('/training.yml') as config_file:
        config_data = yaml.load(config_file)
        set_path    = config_data['set_path']
        model_path  = config_data['model_path']
        lr          = config_data['lr']
    
    # Load the data
    x_train = np.load(os.path.join(set_path, 'x_train.npy'))
    x_val   = np.load(os.path.join(set_path, 'x_val.npy'))
    x_test  = np.load(os.path.join(set_path, 'x_test.npy'))
    t_train = np.load(os.path.join(set_path, 't_train.npy'))
    t_val   = np.load(os.path.join(set_path, 't_val.npy'))
    t_test  = np.load(os.path.join(set_path, 't_test.npy'))
    y_train = np.load(os.path.join(set_path, 'y_train.npy'))
    y_val   = np.load(os.path.join(set_path, 'y_val.npy'))
    y_test  = np.load(os.path.join(set_path, 'y_test.npy'))
    
    # Build the model
    input_image = Input(shape=x_train[0].shape)
    input_time = Input(shape=t_train[0].shape)
    i = Conv2D(filters=5, kernel_size=10, padding='same', activation='relu')(input_image)
    i = Conv2D(filters=1, kernel_size=10, padding='same', activation='relu')(i)
    i = Flatten()(i)
    t = Flatten()(input_time)
    ti = concatenate([i, t])
    ti = Dense(256, activation='relu')(ti)
    ti = Dropout(0.2)(ti)
    outputs = Dense(2, activation='sigmoid')(ti)
    
    model = Model(inputs=[input_image, input_time], outputs=outputs)
    
    model.compile(optimizer='adam',
                 loss=haversine_loss,
                 metrics=[haversine_loss])
    
    # Wrap the model and train
    learner = ktrain.get_learner(model, train_data=([x_train, t_train], y_train),
                                 val_data=([x_val, t_val], y_val))
    
    learner.autofit(lr)
    learner.model.save(os.path.join(model_path, 'new_model.h5'))
    
    # Evaluate
    x_test = np.expand_dims(x_test, 3)
    y_hat = learner.model.predict([x_test, t_test])
    print('\n-----------------Test set performance-----------------------------')
    print(haversine_loss(y_test, y_hat.astype('double')).numpy())


if __name__ == '__main__':
    main()
