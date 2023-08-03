from glob import glob
from ultralytics import YOLO
from testing import localisation_inference


if __name__ == '__main__':
    project_name = 'Testing'
    training_name = '500_epochs'

    ls = glob(f'{project_name}/{training_name}/*.pt')
    for path in ls:
        model = YOLO(path)
        localisation_inference(model, path, training_name)