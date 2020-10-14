# w251-project

<get_skies.sh> starts <get_skies_random.py> but because Stellarium does not quit properly, then have to run <upload_images_s3.py>.  Need to make sure have appropriate AWS S3 config entries in .aws folder.

<get_skies_random.py> uses <ssc_generator.yml> for configuration values (ie lat, long, # of images etc)

<delete_images.py> deletes everything in images folder AND also s3 bucket to create a clean start


