import torch
import torchvision
import os
import json
from PIL import Image
from transform import get_transforms
import numpy as np


def pascalvoc_to_coco(box):
    """
    Convert a bound box (list of length 4) from pascal_voc format to coco format
    :param box: List of length 4 [xmin, ymin, xmax, ymax]
    :return: List of length 4 [xmin, ymin, width, height]
    """
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]


def absolute_to_relative(box, width, height):
    """
    Rescale a bounding box from absolute coordinates to relative (0 to 1) coordinates
    """
    return [box[0] / width, box[1] / height, box[2] / width, box[3] / height]


def relative_to_absolute(box, width, height):
    """
    Rescale a bounding box from relative coordinates to absolute coordinates
    """
    return [box[0] * width, box[1] * height, box[2] * width, box[3] * height]


class BDD100KDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, json_dir, split, valid_proportion=0.1, model='ssd'):
        print(f'Initiating {split} dataset...')
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.split = split
        self.valid_proportion = valid_proportion
        self.model = model
        self.split_filename = 'train' if split == 'valid' else split
        self.size = 300 if model == 'ssd' else 320
        # self.yolo_resize = torchvision.transforms.Resize((320, 320), torchvision.transforms.InterpolationMode.NEAREST)

        assert split in ['train', 'test', 'valid'], 'BDD100KDataset.__init__(): split must be one of "train", "test"'
        assert model in ['ssd', 'yolo'], "model must be one of 'ssd', 'yolo'"

        with open(os.path.join(json_dir, self.split_filename + '.json'), 'r') as infile:
            objects = json.load(infile)

        # separate the dataset into training and valid
        valid_index = int(len(objects) * self.valid_proportion)
        if split == 'train':
            objects = objects[valid_index:]
        elif split == 'valid':
            objects = objects[:valid_index]

        # some images exist in the annotations file but are missing from the image directory
        # do not include these images in training
        # also remove images with no annotations
        self.objects = []
        self.images = []
        for o in objects:
            if os.path.exists(os.path.join(self.image_dir, self.split_filename, o['name'])) and \
                    len(o['labels']):
                self.objects.append(o)
                self.images.append(o['name'])

        with open(os.path.join(json_dir, 'class_index.json'), 'r') as infile:
            self.class_index = json.load(infile)
            self.class_index['background'] = 0

    def __getitem__(self, i):
        image_path = os.path.join(self.image_dir, self.split_filename, self.images[i])
        image = Image.open(image_path).convert('RGB')
        image = np.asarray(image)
        full_labels = self.objects[i]['labels']
        labels = [f['category'] for f in full_labels]
        boxes = [f['box2d'] for f in full_labels]  # default is pascal_voc format
        if not boxes:
            print(i)

        # apply augmentation if in training stage
        if self.split == 'train':
            aug_transform = get_transforms(self.model)
            transformed = aug_transform(image=image, bboxes=boxes, class_labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']

        if self.model == 'yolo':
            boxes = [pascalvoc_to_coco(box) for box in boxes]

        # convert to relative(normalized) coordinates
        boxes = [absolute_to_relative(box, self.size, self.size) for box in boxes]
        # convert to tensors
        image = torchvision.transforms.ToTensor()(image)
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor([self.class_index[l] for l in labels])

        if self.model == 'yolo':
            # image = self.yolo_resize(image)
            targets = torch.cat([torch.zeros(size=(len(boxes), 1)), labels.view(-1, 1), boxes], 1)
            return image_path, image, targets
        else:
            difficulties = torch.zeros(size=labels.size()).byte()
            return image, boxes, labels, difficulties

    def collate_fn(self, batch):
        if self.model == 'ssd':
            images, boxes, labels, difficulties = [], [], [], []
            for b in batch:
                images.append(b[0])
                boxes.append(b[1])
                labels.append(b[2])
                difficulties.append(b[3])
            images = torch.stack(images, dim=0)
            return images, boxes, labels, difficulties
        else:  # yolo
            paths, images, targets = [], [], []
            for b in batch:
                paths.append(b[0])
                images.append(b[1])
                targets.append(b[2])
            images = torch.stack(images, dim=0)
            # assign index to each target
            for i, target in enumerate(targets):
                target[:, 0] = i
            targets = torch.cat(targets)
            return paths, images, targets

    def __len__(self):
        return len(self.images)


def get_train_valid_dataloaders(batch_size,
                                image_dir,
                                json_dir,
                                valid_proportion=0.1,
                                num_workers=2,
                                model='ssd'):
    """
    Returns two data loaders, one for training and one for validation
    """
    train_dataset = BDD100KDataset(image_dir, json_dir, split='train', valid_proportion=valid_proportion, model=model)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size,
                                               num_workers=num_workers,
                                               collate_fn=train_dataset.collate_fn,
                                               shuffle=True, pin_memory=False)

    valid_dataset = BDD100KDataset(image_dir, json_dir, split='valid', valid_proportion=valid_proportion, model=model)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size,
                                               num_workers=num_workers,
                                               collate_fn=valid_dataset.collate_fn,
                                               shuffle=True, pin_memory=False)
    return train_loader, valid_loader


def get_test_dataloader(batch_size,
                        image_dir,
                        json_dir,
                        num_workers=2,
                        model='ssd'):
    """
    Returns a data loader for the test dataset
    """
    dataset = BDD100KDataset(image_dir, json_dir, split='test', model=model)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size,
                                       num_workers=num_workers,
                                       collate_fn=dataset.collate_fn,
                                       shuffle=True, pin_memory=True)
