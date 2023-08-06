from train import training
import os
from testing import download_test_data


def s3_sync(source, destination):
    sync_command = f"aws s3 sync {source} {destination}"
    os.system(sync_command)


def download_training_set():
    s3_image_path = 's3://document-ai-training-data/training_data/table_localisation/column/base_data/'
    local_image_path = 'base_data/'
    s3_sync(s3_image_path, local_image_path)


if __name__ == '__main__':

    project_name = 'RTDETR'
    training_name = 'detr_300'

    download_training_set()
    download_test_data()

    params = {
        'data': "data.yaml",
        'epochs': 300,
        'save_period': 50,
        'batch': 16,
        'single_cls': True,
        'cache': True,
        'project': project_name,
        'name': training_name,
        'lr0': 0.001,
        'lrf': 0.1,
        'close_mosaic': 100,
    }
    training(params, training_name)
    # os.system(f'rm -r {project_name}')
