from glob import glob
from ultralytics import YOLO
from testing import localisation_inference
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
    training_name = 'imgsz_testing'

    # download_training_set()
    download_test_data()

    project_name = 'Testing'
    training_name = 'imgsz3'

    # ls = glob(f'{project_name}/{training_name}/weights/*.pt')
    # for path in ls:
    #     model = YOLO(path)
    #     localisation_inference(model, path, training_name)
    model = YOLO('yolo_v8m_jpg_png.pt')
    localisation_inference(model, project_name, training_name)