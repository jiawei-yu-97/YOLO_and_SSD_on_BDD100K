This repository hosts the code necessary to replicate the experiment results mentioned in our reports. 

The code for YOLOv3 was adapted from https://github.com/eriklindernoren/PyTorch-YOLOv3. <br>
The code for SSD was adapted from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection <br>
We made adjustments to both code bases to be more compatible with our experiments. 

The trained model checkpoint files are available on google drive. Unfortunately Github does not allow any file over 100 MBs. The checkpoint for SSD is at [here](https://drive.google.com/file/d/1TTXsS3E4lIfS_li1UhDlLOCktAGQYxlw/view?usp=sharing), and the checkpoint file for YOLOv3 is at [here](https://drive.google.com/file/d/1xxonizu4fhCSuEfpIqTGuiCSqqKGxV7a/view?usp=sharing). The checkpoint files should be placed in the `checkpoints` folder, for the testing process to work smoothly. <br>

We've included a mini dataset for the purpose of verifying and playing the workflow. Please go into the `datasets` directory and unzip `datasets.zip` there. Once unzipping is done, four folders will be created. The `images_300` and `images_320` folders contain 1000 images each, of size 300x300 and 320x320 respectively, for the SSD and YOLOv3 models. The `labels_300` and `labels_320` contains the json files that describes each image and the bounding boxes for objects within the images. The SSD model will use `images_300` and `labels_300`, and the YOLOv3 will use the other two. <br>
If you want to play around with the full dataset, they are also available on google drive. [This link](https://drive.google.com/file/d/1FMky1cpmXqI3JEdcJF1h97vTeRrwFN4d/view?usp=sharing) contains all images and labels, rescaled to 300x300 for the SSD model. [This link](https://drive.google.com/file/d/1OHnnPtNT3zABCH14AMi5r8ptKkqSOVYe/view?usp=sharing) contains the 320x320 scaled images and labels for the YOLOv3 model. Please unzip them in the `datasets` directory. <br>

For a description of the dataset and to download the original versions, see https://bair.berkeley.edu/blog/2018/05/30/bdd/ . The original resolution of the images is 1280x720. We've resized them so they're easier to work with. <br>

The dataset is divided into train (around 70000 images) and test(10000 images). During training, we further set aside a certain number of frames to be the validation data. <br>
Once you've prepared the checkpoint files and the dataset, to test the performance of the SSD model on the test set, go into the `SSD` directory and run `test_ssd.py`. Similarly, `test_yolo.py` in the `YOLOv3` directory will perform testing using the YOLOv3 model. You can also run `train_ssd.py` or `train_yolo.py` to continue training on our trained model. We've train the YOLOv3 model for 100 epochs, and the SSD model for 65 epochs. If you want to repeat the training process from epoch 0, remove the checkpoint files and run the training scripts. 

If you run into any difficulties feel free to contact me at jiawei.yu@mail.utoronto.ca
