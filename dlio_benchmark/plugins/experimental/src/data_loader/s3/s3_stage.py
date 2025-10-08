import argparse
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error

import data_utilities as du

load_dotenv('dlio.env')
DATASET_FOLDER = os.environ['DATASET_FOLDER'] #'/home/keithpij/dlio_benchmark/data/unet3d' 
BUCKET_NAME = os.environ['BUCKET_NAME']


def put_folder(bucket_name: str, dataset_folder: str, split: str, expand_dataset: int=0) -> int:
    '''
    Removes all objects from the specified bucket.
    '''
    url, access_key, secret_key, secure = du.get_minio_credentials()

    try:
        # Create client with access and secret key
        client = Minio(url,  # host.docker.internal
                    access_key,  
                    secret_key, 
                    secure=secure)

        # Make the bucket if it does not exist.
        found = client.bucket_exists(bucket_name)
        if not found:
            client.make_bucket(bucket_name)
            logging.info(f'Creating {bucket_name}.')
        else:
            logging.info(f'Bucket {bucket_name} already exists.')

        count = 0
        samples_dir = os.path.join(dataset_folder, split)

        if expand_dataset:
            for _ in range(expand_dataset):
                for entry in os.listdir(samples_dir):
                    count += 1
                    sample_file_path = os.path.join(samples_dir, entry)
                    sample_object_path = f'{split}/sample_{count}.npz'
                    client.fput_object(bucket_name, sample_object_path, sample_file_path)
                    if count % 5 == 0:
                        logging.info(f'{count} objects uploaded to {bucket_name}.')
        else:
            for entry in os.listdir(samples_dir):
                count += 1
                sample_file_path = os.path.join(samples_dir, entry)
                sample_object_path = f'{split}/{entry}'
                client.fput_object(bucket_name, sample_object_path, sample_file_path)
                if count % 5 == 0:
                    logging.info(f'{count} objects uploaded to {bucket_name}.')

    except S3Error as s3_err:
        logging.error(f'S3 Error occurred: {s3_err}.')
        raise s3_err
    except Exception as err:
        logging.error(f'Error occurred: {err}.')
        raise err

    return count


def main():
    '''
    Main function that processes the arguments sent to this module via the command line.
    '''
    parser = argparse.ArgumentParser(description='ML Command line interface.')
    parser.add_argument('-lb', '--list_buckets', help='List all buckets.', action='store_true')
    parser.add_argument('-lo', '--list_objects', help='List all objects in the specified bucket.', action='store_true')
    parser.add_argument('-eb', '--empty_bucket', help='Remove all objects in the specified bucket.', action='store_true')
    parser.add_argument('-load', '--load_bucket', help='Load unet3d dataset into the specified bucket.', action='store_true')
    args = parser.parse_args()

    # Set logging level.
    logging.getLogger().setLevel(logging.INFO)
    
    if args.empty_bucket:
        count = du.empty_bucket(BUCKET_NAME)
        print(f'Number of objects removed in {BUCKET_NAME}:', count)
    if args.list_buckets:
        bucket_list = du.get_bucket_list()
        print(bucket_list)
    if args.list_objects:
        object_list = du.get_object_list(BUCKET_NAME, prefix='train/')
        print(f'Number of objects in {BUCKET_NAME}:', len(object_list))
        print(object_list[:10])
    if args.load_bucket:
        put_folder(BUCKET_NAME, DATASET_FOLDER, 'train', 0)


if __name__ == '__main__':
    main()
