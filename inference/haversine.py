"""
Implements a haversine loss function for use in
TensorFlow models
"""

import tensorflow as tf
import numpy as np
import unittest


def haversine_loss(y_true, y_pred, R=3443.92):
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
    lat1  = y_true[:,0]
    lat2  = y_pred[:,0]
    long1 = y_true[:,1]
    long2 = y_pred[:,1]
    
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
    return tf.reduce_mean(tf.square(d))


class TestHaversine(unittest.TestCase):

    def test_isolat(self):
        self.assertTrue(np.abs(haversine_loss(
            np.array([[0, 0]]),
            np.array([[0, 180]])).numpy() - 117059281.6) < 0.1)
    
    def test_isolong(self):
        self.assertTrue(np.abs(haversine_loss(
            np.array([[0, 0]]),
            np.array([[90, 0]])).numpy() - 29264820.4) < 0.1)
        
    def test_vaRota(self):
        self.assertTrue(np.abs(haversine_loss(
            tf.constant([[36.8354, -76.2983]]),
            tf.constant([[36.6237, -6.3601]]),
            R=6378.14).numpy() - 37065212.0) < 0.1)
        
            
if __name__ == "__main__":
    unittest.main()