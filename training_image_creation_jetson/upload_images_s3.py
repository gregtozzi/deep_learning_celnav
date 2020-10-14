'''
Takes all files in stars/images folder and uploads to s3 bucket.
Need credentials from .aws/credentials file to get access to s3 bucket

'''


import boto3
import os
from botocore.exceptions import ClientError

bucket_name='w251-final-project'
image_dir='/w251-project/stars/images/'

def check_file(s3_client, bucket, key):
    '''
    checks if key already in bucket
    returns true if key already in bucket and false if not
    '''
    
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        return int(e.response['Error']['Code']) != 404
    return True




def upload_file(s3_client,file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        print ('File not uploaded:',object_name)
        return False
    print('File uploaded:',object_name)
    return True

#get list of files from image directory
file_list=os.listdir(image_dir)
print('# of files:',len(file_list))
s3_client=boto3.client('s3')

i=1
#cycle through files and upload to s3 bucket if not already there
for object_name in file_list:
    
    file_name=os.path.join(image_dir,object_name)
    
    print (i, object_name)
    i+=1
    
    #checks if file already there and uploads if not
    #note there is a little lag for S3 to register is there
    if not check_file(s3_client, bucket_name, file_name):
        #print(' not there')
        upload_file(s3_client,file_name,bucket_name,object_name)
    else:
        #print('there')
        print ('File already there:',file_name)