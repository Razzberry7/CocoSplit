import os
from enum import Enum

################################################

### Config ###

# Enum for mode
class Mode(Enum):
    SPLIT = 0
    AUGMENT_ONLY = 1

### DEFAULTS ###
mode = -1
dataset_dir = "./data/default/"
num_of_splits = "0"
seed = "-1"
blurFlag = "1"
blurAmount = 0
resizeFlag = "1"
resizeAmount = 100
destination_dir = "./result/"
finished_name = "default_name"

### INPUT ###

# CocoSplit Mode (Splitting & Augmentations | Just augmentations)
mode = int(input("What mode would you like to run CocoSplit in? (0 = Splitting | 1 = Just Augmentations)\n>"))

# Directory of the dataset being used
dataset_dir = input("What is the directory of the dataset being augmented? (ex: ./data/MergeSet30/)\n>")

if Mode[mode] == Mode.SPLIT:
    # Number of splits created from 1 original image
    num_of_splits = input("How many splits should be made per original photo?\n>")

    # Custom seed for random splits
    seed = input("Enter the seed you would like to use for the random splits (if none, input -1):\n>")

# Option to blur images
blurFlag = input("Blur the images? (0 = Yes | 1 = No):\n>")

if blurFlag == 0:
    blurAmount = int(input("What K-size should be used for the blurring? (default is 2)\n>"))

# Option to resize images
resizeFlag = input("Resize the images? (0 = Yes | 1 = No):\n>")

if resizeFlag == 0:
    resizeAmount = int(input("What percentage of the original size should the images be resized to? \n(ex: 100 = 100% of the original size, so no change)\n>"))

# Directory that the finished dir will be sent to + Name
destination_dir = input("Where do you want the finished directory to be sent? (ex: ../../yolov5/data/training_data/)\n>")

# File name of the yolo after conversion
finished_filename = input("What do you want to name the finished directory? (ex: MergeSet30_10s)\n>")


################################################

# Starts the program
os.system("nohup python cocosplit.py " + \
    mode + " " + \
    dataset_dir + " " + \
    num_of_splits + " " + \
    seed + " " + \
    blurFlag + " " + \
    blurAmount + " " + \
    resizeFlag + " " + \
    resizeAmount + " " + \
    destination_dir + " " + \
    finished_filename + " " + " &")


