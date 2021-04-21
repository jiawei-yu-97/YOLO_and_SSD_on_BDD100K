from __future__ import division

import sys
from os import path
import os
import json
import datetime
from time import time

import torch
from terminaltables import AsciiTable
from torch.autograd import Variable
import torch.optim as optim
from torchsummary import summary

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from DataLoader import get_train_valid_dataloaders, get_test_dataloader

from models import load_model
from utils.loss import compute_loss
from test import _evaluate
from utils.utils import to_cpu

#################
# TRAINING CONFIG
#################
model = 'config/yolov3.cfg'
epochs = 200
n_worker = 2
pretrained_weights = 'weights/yolov3.weights'
checkpoint_interval = 1
evaluation_interval = 1
iou_thres = 0.45
conf_thres = 0.01
nms_thres = 0.5
logdir = 'logs'
verbose = True
valid_proportion = 0.1
batch_size = 16
print_frequency = 500
eval_frequency = 2000

# Model Hyper Parameters
learning_rate = 0.001
lr_steps = [(80000, 0.5), (160000, 0.2), (240000, 0.2), (360000, 0.2)]


###################
# LOGGING FUNCTIONS
###################
def log_and_print(log_file, message):
    with open(log_file, 'a') as outfile:
        outfile.write(message + '\n')
    print(message)


def log_and_print_eval(log_file, metrics_output, class_names):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output

        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            ap_class_dict = {c: i for i, c in enumerate(ap_class)}
            #for i, c in enumerate(ap_class):
            for c in range(1, 14):
                if c in ap_class_dict:
                    i = ap_class_dict[c]
                    ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
                else:
                    ap_table += [[c, class_names[c], "%.5f" % 0]]
            log_and_print(log_file, AsciiTable(ap_table).table)
        log_and_print(log_file, f"---- mAP {AP.sum()/13:.5f} ----")
    else:
        log_and_print(log_file, "---- mAP not measured (no detections found by model) ----")


def format_timelapsed(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f'{seconds} seconds'
    elif seconds < 3600:
        return f'{seconds // 60} minutes {seconds % 60} seconds'
    else:
        hours = seconds // 3600
        remaining = seconds - 3600 * hours
        minutes = remaining // 60
        return f'{hours} hours {minutes} minutes {remaining % 60} seconds'


if __name__ == '__main__':
    start = time()

    IMAGE_DIR = "../datasets/images_320"
    JSON_DIR = "../datasets/labels_320"
    with open('../datasets/labels_320/class_index.json') as f:
        label_map = json.load(f)
    label_map['background'] = 0
    class_names = {item: key for (key, item) in label_map.items()}
    class_names = [class_names[i] for i in range(len(label_map))]

    checkpoint_dir = '../checkpoints'
    checkpoint_file = 'checkpoints_yolov3.pth'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ############
    # Create model
    # ############
    now = datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
    if not os.path.exists(checkpoint_path):
        try:
            model = load_model(model, pretrained_weights)
        except Exception as e:
            print(f'Model weight file {pretrained_weights} does not exist. Please run weights/download_weights.sh to download weights file')
            raise e
        model.hyperparams['height'] = 320
        model.hyperparams['width'] = 320
        start_epoch = 0

        # create log file
        now = datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
        log_file = f'../logs/{now}.txt'

        if verbose:
            summary(model, input_size=(3, model.hyperparams['width'], model.hyperparams['height']))
    else:
        checkpoint = torch.load(checkpoint_path)
        weights = checkpoint['model']
        model = load_model(model)
        model.load_state_dict(weights)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        log_file = checkpoint['log_file']
        if not os.path.exists(log_file):
            log_file = f'../logs/{now}.txt'


    model.hyperparams['height'] = 320
    model.hyperparams['width'] = 320
    model.hyperparams['learning_rate'] = learning_rate
    model.hyperparams['lr_steps'] = lr_steps
    if verbose:
        print(model.hyperparams)


    log_and_print(log_file, f'Using {device}')

    # #################
    # Create Dataloader
    # #################
    train_loader, valid_loader = get_train_valid_dataloaders(batch_size,
                                                             image_dir=IMAGE_DIR,
                                                             json_dir=JSON_DIR,
                                                             valid_proportion=valid_proportion,
                                                             num_workers=n_worker,
                                                             model='yolo')
    test_loader = get_test_dataloader(batch_size, IMAGE_DIR, JSON_DIR, model='yolo')

    # ################
    # Create optimizer
    # ################
    params = [p for p in model.parameters() if p.requires_grad]

    if (model.hyperparams['optimizer'] in [None, "adam"]):
        optimizer = torch.optim.Adam(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
        )
    elif (model.hyperparams['optimizer'] == "sgd"):
        optimizer = torch.optim.SGD(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum'])
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")


    for epoch in range(start_epoch, epochs):
        print("\n---- Training Model ----")

        for batch_i, (_, imgs, targets) in enumerate(train_loader):
            model.train()
            batches_done = len(train_loader) * epoch + batch_i

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            outputs = model(imgs)
            loss, loss_components = compute_loss(outputs, targets, model)
            loss.backward()

            ###############
            # Run optimizer
            ###############
            if batches_done % model.hyperparams['subdivisions'] == 0:
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = model.hyperparams['learning_rate']
                if batches_done < model.hyperparams['burn_in']:
                    # Burn in
                    lr *= (batches_done / model.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value
                # Set learning rate
                for g in optimizer.param_groups:
                    g['lr'] = lr
                # Run optimizer
                optimizer.step()
                # Reset gradients
                optimizer.zero_grad()

            # ############
            # Log progress
            # ############
            log_str = ""
            log_str += AsciiTable(
                [
                    ["Type", "Value"],
                    ["IoU loss", float(loss_components[0])],
                    ["Object loss", float(loss_components[1])],
                    ["Class loss", float(loss_components[2])],
                    ["Loss", float(loss_components[3])],
                    ["Batch loss", to_cpu(loss).item()],
                ]).table

            if batch_i % print_frequency == 0:
                if verbose:
                    log_and_print(log_file,
                                  f'\nEpoch: {epoch}; Batch: {batch_i}; images:{batch_i * batch_size}; {format_timelapsed(time() - start)} elapsed')
                    log_and_print(log_file, f'Learning Rate: {lr}')
                    log_and_print(log_file, log_str)

                # Tensorboard logging
                '''
                tensorboard_log = [
                    ("train/iou_loss", float(loss_components[0])),
                    ("train/obj_loss", float(loss_components[1])),
                    ("train/class_loss", float(loss_components[2])),
                    ("train/loss", to_cpu(loss).item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)
                '''

            model.seen += imgs.size(0)

            # Evaluate
            if batch_i % eval_frequency == 0 and batch_i > 0:
                print("\n---- Evaluating Model ----")
                # Evaluate the model on the validation set
                metrics_output = _evaluate(
                    model,
                    valid_loader,
                    class_names,
                    img_size=model.hyperparams['height'],
                    iou_thres=iou_thres,
                    conf_thres=conf_thres,
                    nms_thres=nms_thres,
                    verbose=False
                )

                log_and_print_eval(log_file, metrics_output, class_names)

        # #############
        # Save progress
        # #############

        # Save model to checkpoint file for every end of epoch
        print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'log_file': log_file},
                   checkpoint_path)

        ##########
        # Evaluate
        ##########

        if epoch % evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = _evaluate(
                model,
                valid_loader,
                class_names,
                img_size=model.hyperparams['height'],
                iou_thres=iou_thres,
                conf_thres=conf_thres,
                nms_thres=nms_thres,
                verbose=False
            )
            log_and_print_eval(log_file, metrics_output, class_names)

            '''
            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                    ("validation/precision", precision.mean()),
                    ("validation/recall", recall.mean()),
                    ("validation/mAP", AP.mean()),
                    ("validation/f1", f1.mean())]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)
            '''