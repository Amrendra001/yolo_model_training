from ultralytics import YOLO
from testing import localisation_inference, download_test_data


if __name__ == '__main__':
    download_test_data()

    for iou in range(1, 10):
        iou /= 10
        model = YOLO('pypdfium.pt')
        localisation_inference(model, f'conf = {iou}', 'pypdfium_conf_oldconv_testimg', iou)