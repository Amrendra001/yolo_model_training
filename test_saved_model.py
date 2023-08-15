from glob import glob
from ultralytics import YOLO
from testing import localisation_inference


if __name__ == '__main__':

    project_name = 'Testing_new_2'
    training_name = 'yolov8l_augment2'

    ls = glob(f'{project_name}/{training_name}/weights/*.pt')
    for path in ls:
        model = YOLO(path)
        localisation_inference(model, path, training_name)