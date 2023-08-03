from train import training
import os
from testing import download_test_data


def s3_sync(source, destination):
    sync_command = f"aws s3 sync {source} {destination}"
    os.system(sync_command)


def download_training_set():
    s3_image_path = 's3://document-ai-training-data/training_data/table_localisation/column/new_smaller_training_data/'
    local_image_path = 'new_smaller_training_data/'
    s3_sync(s3_image_path, local_image_path)


if __name__ == '__main__':

    project_name = 'Testing'
    training_name = '500_epochs'

    download_training_set()
    download_test_data()

    params = {
        'data': "data.yaml",
        'epochs': 500,
        'save_period': 100,
        'batch': 32,
        'single_cls': True,
        'cache': True,
        'project': project_name,
        'name': training_name,
    }
    training(params)
    # os.system(f'rm -r {project_name}')
