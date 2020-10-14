import numpy as np


def get_file_name(lat, long, date):
    """
    Converts the lat, long, and date associated
    with a screenshot to a unique file name.
    """
    file_name = str(lat) + '+' + str(long) + '+' + date
    return file_name


def timelinspace(start, end, steps):
    """
    Given a starting time and an ending time,
    generate a grid of equally-spaced intermediate
    times.
    
    Args:
        start: str - Must be able to be converted
               to an np.datetime64 object
        stop:  str - Must be able to be converted
               to an np.datetime64 object
        steps: int - Sets the number of intervals
               between start and stop
               
    Returns:   list of times converted to strings
    """
    # Convert to datetime64 to make calculations
    # easier
    start_dt = np.datetime64(start)
    end_dt   = np.datetime64(end)
    
    time_diff = end_dt - start_dt
    delta_t = time_diff / steps
    
    step_vec = np.arange(0, steps + 1)
    times = start_dt + step_vec * delta_t
    
    return [str(time) for time in times]