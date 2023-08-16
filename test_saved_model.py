from glob import glob
from ultralytics import YOLO, RTDETR
from testing import localisation_inference


if __name__ == '__main__':

    project_name = 'Testing_new_2'
    training_name = 'DETR'

    ls = glob(f'DETR/weights/*.pt')
    for path in ls:
        model = RTDETR(path)
        localisation_inference(model, path, training_name)