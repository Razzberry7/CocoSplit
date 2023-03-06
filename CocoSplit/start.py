import os
################################################

### Config ###


# Directory of the dataset to split
dataset_dir = input("What is the dir of the dataset being split? (ex: ./data/MergeSet30/)\n")

# Directory that the finished dir will be sent to
destination_dir = input("Where do you want the finished directory to be sent? (ex: ../../yolov5/data/weights/)\n")

# File name of the yolo after conversion
finished_filename = input("What do you want to name the finished directory? (ex: MergeSet30_10s)\n")

# Number of splits created from 1 original
num_of_splits = input("How many splits should be made per original photo?\n")

# Custom seed for random splits
seed = input("Enter the seed you would like to use for the random splits (if none, input -1):\n")

# Custom seed for random splits
blurFlag = input("Blur the pictures? (Y/N):\n")

################################################

# Starts the program
os.system("nohup python cocosplit.py " + dataset_dir + " " + destination_dir + " " + finished_filename + " " + num_of_splits + " " + seed + " " + blurFlag + " &")


