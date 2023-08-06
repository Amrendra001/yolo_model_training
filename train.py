# import wandb
from ultralytics import YOLO, RTDETR
from testing import localisation_inference


def training(params, training_name):
    model = RTDETR("RTDETR/detr_3002/weights/last.pt")
    model.train(**params)
    localisation_inference(model, params, training_name)