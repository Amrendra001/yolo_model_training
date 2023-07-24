# import wandb
from ultralytics import RTDETR
import os


def s3_sync(source, destination):
    sync_command = f"aws s3 sync {source} {destination}"
    os.system(sync_command)


if __name__ == '__main__':
        s3_image_path = 's3://document-ai-training-data/training_data/table_localisation/column/base_data/'
        local_image_path = 'base_data/'
        s3_sync(s3_image_path, local_image_path)

        model = RTDETR("rtdetr-l.pt")
        model.train(data="data.yaml", epochs=5, save_period=10, val=True, project='RTDETR_Training', batch=16, amp=True, single_cls=True, augment=False, cache='ram')
