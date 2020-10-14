# delete all files in w251-project s3 bucket and also in the local image folder
import boto3
import os
import yaml

def empty_s3_bucket(bucket_name):
  client = boto3.client('s3')
  response = client.list_objects_v2(Bucket=bucket_name)
  if 'Contents' in response:
    for item in response['Contents']:
      print('deleting file', item['Key'])
      client.delete_object(Bucket=bucket_name, Key=item['Key'])
      while response['KeyCount'] == 1000:
        response = client.list_objects_v2(
          Bucket=bucket_name,
          StartAfter=response['Contents'][0]['Key'],
        )
        for item in response['Contents']:
          print('deleting file', item['Key'])
          client.delete_object(Bucket=bucket_name, Key=item['Key'])


with open('ssc_generator.yml') as config_file:
  config_data = yaml.load(config_file)
    
bucket_name = config_data['aws_s3_bucket']
image_dir = config_data['image_path']

print ('emptying s3 bucket:', bucket_name)
empty_s3_bucket(bucket_name)

print ('removing image files:', image_dir)
file_list=os.listdir(image_dir)
for file_name in file_list:
  file_delete=os.path.join(image_dir,file_name)
  os.remove(file_delete)
