# import wandb
import os
from ultralytics import RTDETR, YOLO

def s3_sync(source, destination):
    sync_command = f"aws s3 sync {source} {destination}"
    os.system(sync_command)


if __name__ == '__main__':
        s3_image_path = 's3://document-ai-training-data/training_data/table_localisation/column/base_data/'
        local_image_path = 'base_data/'
        s3_sync(s3_image_path, local_image_path)

        model = RTDETR("rtdetr-l.pt")
        # model = YOLO('yolov8m.pt')
        model.train(data="data.yaml", epochs=3, batch=16, single_cls=True, cache=True)
