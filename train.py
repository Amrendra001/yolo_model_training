import wandb
from ultralytics import YOLO
from testing import localisation_inference


def training(params):
    model = YOLO('yolov8s.pt')
    model.train(**params)
    localisation_inference(model, params)