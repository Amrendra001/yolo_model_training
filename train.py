# import wandb
from ultralytics import YOLO
from testing import localisation_inference


def training(params, training_name):
    # model = YOLO('yolov8s.pt')
    model = YOLO('Testing_new_2/0_to_1002/weights/best.pt')
    model.train(**params)
    localisation_inference(model, params, training_name)