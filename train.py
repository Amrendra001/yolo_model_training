import wandb
from ultralytics import YOLO
from testing import localisation_inference


def training(params, training_name):
    model = YOLO('yolov8s.pt')
    model.train(**params)
    localisation_inference(model, params, training_name)