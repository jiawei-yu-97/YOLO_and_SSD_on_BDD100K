from tqdm import tqdm
from pprint import PrettyPrinter
import datetime

from utils import *
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from DataLoader import BDD100KDataset, get_test_dataloader


# Parameters
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 16
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = '../checkpoints/checkpoint_ssd300.pth'
min_score = 0.01
max_overlap = 0.45
top_k = 200

# # Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint_path)
model = checkpoint['model']
model = model.to(device)

# # Switch to eval mode
model.eval()

# Load test data
IMAGE_DIR = "../datasets/images_300"
JSON_DIR = "../datasets/labels_300"

test_loader = get_test_dataloader(batch_size, IMAGE_DIR, JSON_DIR, num_workers=workers, model='ssd')

# create log file
log_file = f'../logs/ssd_test.txt'

###################
# LOGGING FUNCTIONS
###################
def log_and_print(log_file, message):
    with open(log_file, 'a') as outfile:
        outfile.write(message + '\n')
    print(message)


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


if __name__ == '__main__':
    evaluate(test_loader, model)