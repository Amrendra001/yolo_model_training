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

    project_name = 'ultralytics_8_0_157'
    training_name = 'yolov8s_ultralytics_latest'

    download_training_set()
    download_test_data()

    params = {
        'data': "data.yaml",
        'epochs': 100,
        'save_period': 25,
        'batch': 32,
        'single_cls': True,
        'cache': 'ram',
        'project': project_name,
        'name': training_name,
        'lr0': 0.001,
        'lrf': 0.1,
        'mosaic': 0.1,
        'augment': True,
        'scale': 0.2,
        'fliplr': 0.25,
        'val': True,
    }
    training(params, training_name)
    # os.system(f'rm -r {project_name}')
