import json
import torch
import os
import sys
from os import path
import time

from test import _evaluate
from models import load_model
from train_yolo import log_and_print_eval, log_and_print

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from DataLoader import get_test_dataloader


def test_model(log_file, model, test_loader, class_names, iou_thres, conf_thres, nms_thres):
    print("\n---- Evaluating Model ----")
    # Evaluate the model on the validation set
    metrics_output = _evaluate(
        model,
        test_loader,
        class_names,
        img_size=model.hyperparams['height'],
        iou_thres=iou_thres,
        conf_thres=conf_thres,
        nms_thres=nms_thres,
        verbose=False
    )

    log_and_print_eval(log_file, metrics_output, class_names)


if __name__ == '__main__':
    model = 'config/yolov3.cfg'
    checkpoint_dir = '../checkpoints'
    checkpoint_file = 'checkpoints_yolov3.pth'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    checkpoint = torch.load(checkpoint_path)
    weights = checkpoint['model']
    epochs = checkpoint['epoch']
    model = load_model(model)
    model.load_state_dict(weights)
    print(f'\nLoaded checkpoint from epoch {epochs}.\n')

    IMAGE_DIR = "../datasets/images_320"
    JSON_DIR = "../datasets/labels_320"
    with open('../datasets/labels_320/class_index.json') as f:
        label_map = json.load(f)
    label_map['background'] = 0
    class_names = {item: key for (key, item) in label_map.items()}
    class_names = [class_names[i] for i in range(len(label_map))]

    iou_thres = 0.45
    conf_thres = 0.01
    nms_thres = 0.5

    model.hyperparams['height'] = 320
    model.hyperparams['width'] = 320

    log_file = '../logs/yolo_test.txt'
    with open(log_file, 'w') as outfile:
        outfile.write('')

    test_loader = get_test_dataloader(16, IMAGE_DIR, JSON_DIR, model='yolo')
    test_model(log_file, model, test_loader, class_names, iou_thres, conf_thres, nms_thres)
    log_and_print(log_file, f'----------')

