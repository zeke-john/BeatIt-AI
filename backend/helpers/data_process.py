# NOTE: code to combine all files in the artists folder into a single training data folder
# import os
# import shutil
# import random

# def combine_files_flat(source_dir, output_dir='/Users/zekejohn/Documents/Github/BeatIt-AI/backend/data/training_data'):
#     os.makedirs(output_dir, exist_ok=True)

#     for root_dir, _, files in os.walk(source_dir):
#         for file in files:
#             full_path = os.path.join(root_dir, file)
#             base_name, ext = os.path.splitext(file)
#             destination = os.path.join(output_dir, file)

#             # If file already exists, add random number
#             while os.path.exists(destination):
#                 rand_num = random.randint(1000, 9999)
#                 new_name = f"{base_name}_{rand_num}{ext}"
#                 destination = os.path.join(output_dir, new_name)

#             shutil.copy2(full_path, destination)

# # Your actual path
# source_path = '/Users/zekejohn/Documents/Github/BeatIt-AI/backend/data/artists'
# combine_files_flat(source_path)


# NOTE: code to count number of files in the training data folder, and the number of hours of audio
# import os

# def count_files_in_training_data(training_data_dir):
#     return len(os.listdir(training_data_dir))

# def count_hours_of_audio(training_data_dir):
#     total_hours = 0
#     for file in os.listdir(training_data_dir):
#         if file.endswith('.mp3'):
#             total_hours += os.path.getsize(os.path.join(training_data_dir, file)) / (1024 * 1024 * 1024)
#     return total_hours

# print(count_files_in_training_data('/Users/zekejohn/Documents/Github/BeatIt-AI/backend/data/training_data'))
# print(count_hours_of_audio('/Users/zekejohn/Documents/Github/BeatIt-AI/backend/data/training_data'))


# NOTE: CODE TO UPLOAD TRAINING DATA FOLDER TO AWS S3 IN THE beatit-ai-training-data bucket: /Users/zekejohn/Documents/Github/BeatIt-AI/backend/data/artists
# import os
# import boto3
# from botocore.exceptions import ClientError

# # Use your 'personal' profile
# session = boto3.Session(profile_name='personal')
# s3 = session.client('s3')
# sts = session.client('sts')
# region = session.region_name or 'us-east-1'

# bucket_name = 'beatit-ai-training-data'
# local_folder = '/Users/zekejohn/Documents/Github/BeatIt-AI/backend/data/training_data'
# s3_folder_prefix = 'training-data/'

# identity = sts.get_caller_identity()
# print("👤 AWS Account ID:", identity['Account'])
# print("🧑‍💼 IAM User/Role ARN:", identity['Arn'])
# print("🔐 User ID:", identity['UserId'])


# # === Bucket Creation ===
# def create_bucket_if_not_exists(bucket, region):
#     try:
#         s3.head_bucket(Bucket=bucket)
#         print(f"✅ Bucket '{bucket}' already exists.")
#     except ClientError as e:
#         error_code = int(e.response['Error']['Code'])
#         if error_code == 404:
#             print(f"🪣 Bucket '{bucket}' does not exist. Creating...")
#             if region == 'us-east-1':
#                 s3.create_bucket(Bucket=bucket)
#             else:
#                 s3.create_bucket(
#                     Bucket=bucket,
#                     CreateBucketConfiguration={'LocationConstraint': region}
#                 )
#             print(f"✅ Bucket '{bucket}' created.")
#         else:
#             raise

# # === Folder Upload ===
# def upload_folder_to_s3(local_path, bucket, s3_prefix=''):
#     for root, _, files in os.walk(local_path):
#         for file in files:
#             local_file_path = os.path.join(root, file)
#             relative_path = os.path.relpath(local_file_path, local_path)
#             s3_key = os.path.join(s3_prefix, relative_path).replace('\\', '/')

#             print(f'⬆️ Uploading {local_file_path} to s3://{bucket}/{s3_key}')
#             s3.upload_file(local_file_path, bucket, s3_key)

# # === Execute ===
# create_bucket_if_not_exists(bucket_name, region)
# upload_folder_to_s3(local_folder, bucket_name, s3_folder_prefix)


