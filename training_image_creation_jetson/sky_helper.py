from geopy.distance import geodesic


def get_file_name(lat,long,date):
    file_name=str(lat)+"+"+str(long)+"+"+date
    #print (file_name)
    return file_name



# def next_lat(this_lat,this_long,step_meters):
#     '''
#     Takes current lat/long and step in meters and returns the next lat
#     '''
#     lat_test_slice=.02
#     point1=(this_lat,this_long)
#     point2=(this_lat+lat_test_slice,this_long)
    
#     #this gives you meters for a test_slice
#     meters_test_slice=geodesic(point1,point2).meters
#     #print('meters for slice:',meters_test_slice)
    
#     lat_per_meter=lat_test_slice/meters_test_slice
#     #print('lat per meter:',lat_per_meter)
    
#     next=this_lat+(lat_per_meter*step_meters)
#     #print ('next lat:',next)
#     return next

# def next_long(this_lat, this_long,step_meters):
#     '''
#     Takes current lat/long and step in meters and returns the next long
#     '''
#     long_test_slice=.02
#     point1=(this_lat,this_long)
#     point2=(this_lat,this_long+long_test_slice)
    
#     #this gives you meters for a test_slice
#     meters_test_slice=geodesic(point1,point2).meters
#     #print('meters for slice:',meters_test_slice)
    
#     long_per_meter=long_test_slice/meters_test_slice
#     #print('lat per meter:',long_per_meter)
    
#     next=this_long+(long_per_meter*step_meters)
#     #print ('next long:',next)
#     return next

