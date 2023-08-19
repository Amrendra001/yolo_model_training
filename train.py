# import wandb
from ultralytics import YOLO
from testing import localisation_inference


def training(params, training_name):
    model = YOLO('yolov8m.pt')
    # model = YOLO('Testing_new_2/100_to_200/weights/best.pt')
    model.train(**params)
    localisation_inference(model, params, training_name)