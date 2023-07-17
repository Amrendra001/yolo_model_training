import wandb
from ultralytics import YOLO
import os


def s3_sync(source, destination):
    sync_command = f"aws s3 sync {source} {destination}"
    os.system(sync_command)


if __name__ == '__main__':
        s3_image_path = 's3://document-ai-training-data/training_data/table_localisation/column/png_jpg_data/'
        local_image_path = 'png_jpg_data/'
        s3_sync(s3_image_path, local_image_path)

        model = YOLO("yolov8m.pt")
        model.train(data="data.yaml", epochs=75, save_period=10, val=True, project='Training', batch=32, cache='ram', close_mosaic=10, single_cls=True, augment=True)
        model.val(data="data.yaml")