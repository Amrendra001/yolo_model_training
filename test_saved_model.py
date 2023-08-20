from glob import glob
from ultralytics import YOLO
from testing import localisation_inference, download_test_data


if __name__ == '__main__':
    download_test_data()

    project_name = 'ultralytics_8_0_90'
    training_name = 'yolov8l'

    ls = glob(f'{project_name}/{training_name}/weights/*.pt')
    for path in ls:
        model = YOLO(path)
        localisation_inference(model, path, training_name)