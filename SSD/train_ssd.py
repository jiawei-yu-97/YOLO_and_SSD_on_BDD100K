import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from pprint import PrettyPrinter
from tqdm import tqdm
import datetime

from model import SSD300, MultiBoxLoss
from utils import *
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from DataLoader import BDD100KDataset, get_test_dataloader, get_train_valid_dataloaders

# Data parameters
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
batch_size = 16  # batch size
iterations = 120000  # number of iterations to train
workers = 2  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
valid_proportion = 0.1
min_score = 0.01
max_overlap = 0.45
top_k = 200

cudnn.benchmark = True

# directory of relavant files
checkpoint_path = '../checkpoints/checkpoint_ssd300.pth'
IMAGE_DIR = "../datasets/images_300"
JSON_DIR = "../datasets/labels_300"


###################
# LOGGING FUNCTIONS
###################
def log_and_print(log_file, message):
    with open(log_file, 'a') as outfile:
        outfile.write(message + '\n')
    print(message)


# train SSD model
def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            log_and_print(log_file,
                          'Epoch: [{0}][{1}/{2}]\t'
                          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))


def evaluate(test_loader, model):
    """
    Evaluate.
    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=min_score,
                                                                                       max_overlap=max_overlap,
                                                                                       top_k=top_k)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    # Print AP for each class
    with open(log_file, 'a') as outfile:
        pp = PrettyPrinter(stream=outfile)
        pp.pprint(APs)
    pp = PrettyPrinter()
    pp.pprint(APs)

    log_and_print(log_file, '\nMean Average Precision (mAP): %.3f' % mAP)


def save_checkpoint(epoch, model, optimizer, log_file):
    state = {'epoch': epoch, 'model': model, 'optimizer': optimizer, 'log_file': log_file}
    filename = checkpoint_path
    torch.save(state, filename)


def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint_path, decay_lr_at, log_file

    # Initialize model or load checkpoint
    now = datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
    if not os.path.exists(checkpoint_path):
        # create log file
        log_file = f'../logs/{now}.txt'
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        log_file = checkpoint['log_file']
        if not os.path.exists(log_file):
            log_file = f'../logs/{now}.txt'
        log_and_print(log_file, '\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']


    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = BDD100KDataset(IMAGE_DIR, JSON_DIR, split='train', valid_proportion=valid_proportion)
    train_loader, val_loader = get_train_valid_dataloaders(batch_size, valid_proportion=valid_proportion,
                                                           image_dir=IMAGE_DIR, json_dir=JSON_DIR,
                                                           num_workers=4, model='ssd')

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    epochs = iterations // (len(train_dataset) // 32)
    decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]

    # Epochs
    for epoch in range(start_epoch, epochs+10):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        evaluate(val_loader, model)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, log_file)

if __name__ == '__main__':
    main()