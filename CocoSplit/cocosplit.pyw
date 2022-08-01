### Imports ###
import shutil
import sys

import cv2
import json
import os
import zipfile
import random
import coco_to_yolo
################################################

### Variables ###

# List of the splits
split_list = []

# List of the new annotations correlating to each split
new_annotation_list = []

# List of the original images (used to keep track of what split belongs to what base image)
image_annotation_lists = []

# List of the splits' origins (needed for adjusting annotations)
split_origins = []


# Directory of the dataset to split (passed in from start.py)
dataset_dir = sys.argv[1]

# Directory that the finished directory will be sent to (passed in from start.py)
destination_dir = sys.argv[2]

# Name of the finished directory (passed in from start.py)
destination_dir = destination_dir + sys.argv[3] + "/"

# Number of splits per image (passed in from start.py)
num_of_splits = sys.argv[4]

################################################

###### CODE STARTS HERE #######

# Unzip roboflow zip in ./presplit folder
path = dataset_dir
files = os.listdir(path)
for file in files:
    if file.endswith(".zip"):
        filePath = path + "/" + file
        zip_file = zipfile.ZipFile(filePath)
        for names in zip_file.namelist():
            zip_file.extract(names, path)
        zip_file.close()

# Open the original JSON file, store it as a variable, and close the file
file = open(dataset_dir + 'train/_annotations.coco.json')
old_coco_data = json.load(file)
file.close()

# Create the new JSON file with a framework (info can be changed here if you want)
new_coco_data = \
    {
        "info": {
            "year": "2022",
            "version": "1",
            "description": "Exported from roboflow.ai",
            "contributor": "",
            "url": "https://app.roboflow.ai/datasets/blueberries_moore_drone_6-16-2021/1",
            "date_created": "2022-01-05T20:45:25+00:00"
        },
        "licenses": [
            {
                "id": 1,
                "url": "",
                "name": "Unknown"
            }
        ],
        "categories": [
            {
                "id": 0,
                "name": "berries",
                "supercategory": "none"
            },
            {
                "id": 1,
                "name": "blue",
                "supercategory": "berries"
            },
            {
                "id": 2,
                "name": "green",
                "supercategory": "berries"
            }
        ],
        "images": [],
        "annotations": []
    }

# Method to sort the annotations by what image they belong to (original JSON is unsorted)
def sort_annotations():

    print("Sorting annotations...")

    # Dictionary of annotations from original coco file
    annotations = old_coco_data['annotations']

    # Sort annotations by image_id and store it
    sorted_annotations = sorted(annotations, key=lambda x: x['image_id'])

    # Call next step and pass in sorted_annotations list to be separated
    split_sorted_annotations(sorted_annotations)

# Method to split the annotations up by which image they correspond to
def split_sorted_annotations(sorted_annotations):

    print("Separating sorted annotations by original image...")

    # Dictionary of images from original coco file
    images = old_coco_data['images']

    # Access global var
    global image_annotation_lists

    # Create as many lists as there are original images and store that in a list
    for n in range(len(images)):
        image_annotation_lists.append([])

    # Store the annotations corresponding to the list of image lists
    i = 0
    j = 0
    while i < len(sorted_annotations):
        if sorted_annotations[i]['image_id'] == j:
            image_annotation_lists[j].append(sorted_annotations[i])
            i = (i + 1)
        else:
            j = (j + 1)

    # Call the next step to randomly split the images
    random_split()

# Method to randomly split the original images
def random_split():

    print("Splitting original images...")

    # Dictionary of images from original coco file
    images = old_coco_data['images']

    # Create a new folder (if one doesn't exist already)
    if not os.path.exists("./splits"):
        os.mkdir("./splits")

    # Delete all existing files in that folder (if any exist)
    directory = 'splits'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        os.remove(f)

    # Access global var - split origins (x,y)
    global split_origins

    # Nested while loop that splits images
    i = 0
    while i < len(os.listdir('./presplit/train')) - 1:

        # Import image
        original_img = cv2.imread('./presplit/train/' + images[i]["file_name"])

        # List of splits for each picture
        splits = []

        j = 0
        while j < num_of_splits:

            # Replicate robosplit by creating splits made of half the dimensions of original image
            new_height = int(images[i]['height'] / 2)
            new_width = int(images[i]['width'] / 2)

            # Generate random origin (within bounds)
            random_y = random.randint(0, (images[i]['height'] - new_height))
            random_x = random.randint(0, (images[i]['width'] - new_width))

            # Crop the original image to create the split using the above values
            new_split = original_img[random_y:(random_y + new_height), random_x:(random_x + new_width)]

            # Append the split
            splits.append(new_split)

            # Append the split origins (we'll need this information later)
            split_origins.append([random_x, random_y])

            # Filename of each split
            file_name = './splits/' + str(i) + '_' + str(j) + '_' + str(new_width) + 'x' + str(new_height) + '.jpg'

            # Saving the split
            cv2.imwrite(file_name, splits[j])

            #### Writing the images to the JSON file ####

            # Create the id var
            id = hash(str(i) + "_" +  str(j))

            # Create the data_captured var
            date_captured = images[i]['date_captured']

            # Write new split image to the new JSON file (see helper method for more details)
            write_new_image(id, 1, file_name, new_height, new_width, date_captured, split_list)

            j = (j + 1)

        i = (i + 1)

    # Call next step to adjust the annotations
    adjust_random_annotations()

# Method to adjust the annotations for each split
def adjust_random_annotations():

    print("Adjusting annotations for each split...")

    # Access global vars
    global image_annotation_lists, split_origins

    # Store original images as a var
    original_images = old_coco_data['images']

    # Store copy of global var
    split_images = split_list

    split_annotations = new_annotation_list

    # Original_image index for grabbing the annotations list corresponding to the original image
    original_image = 0

    # These indices have more specific names to not get lost
    i_original_image = 0
    j_split_image = 0
    k_blueberry = 0

    # for loop for the number of original pictures
    for i_original_image in range(len(original_images)):

        # Grab the list of annotations for a given original image
        annotation_list = image_annotation_lists[i_original_image]

        # for loop for the number of split images
        for j_split_image in range(num_of_splits):

            # Calculate the current index of the split we're on
            current_index = i_original_image * num_of_splits + j_split_image

            # Grab that split & origin
            curr_split = split_images[current_index]
            curr_split_origin = split_origins[current_index]

            # for loop for the number of blueberries
            for k_blueberry in range(len(image_annotation_lists[i_original_image])):

                # Grab the old annotation's bounding box and store it
                old_annot = annotation_list[k_blueberry]['bbox']

                # Store all the values in the old_annot bounding box
                old_x = int(old_annot[0])
                old_y = int(old_annot[1])
                old_width = int(old_annot[2])
                old_height = int(old_annot[3])

                # New id
                id1 = len(new_annotation_list)

                # New image_id
                image_id1 = hash(str(i_original_image) + "_" + str(j_split_image))

                # New category_id
                category_id = annotation_list[k_blueberry]['category_id']

                # New origin for original annotation
                new_x1 = old_x - curr_split_origin[0]
                new_y1 = old_y - curr_split_origin[1]

                # New width
                if (new_x1 < curr_split['width']):
                    if (new_x1 + old_width > curr_split['width']):
                        new_width1 = curr_split['width'] - new_x1
                    else:
                        new_width1 = old_width
                else:
                    continue

                # New height
                if (new_y1 < curr_split['height']):
                    if (new_y1 + old_height > curr_split['height']):
                        new_height1 = curr_split['height'] - new_y1
                    else:
                        new_height1 = old_height
                else:
                    continue

                # New bbox list
                new_bbox1 = [new_x1, new_y1, new_width1, new_height1]

                # New area
                new_area1 = new_width1 * new_height1

                # Write to JSON file (see helper method for more detail)
                write_new_annotation(id1, image_id1, category_id, new_bbox1, new_area1, [], 0, new_annotation_list)

                k_blueberry = (k_blueberry + 1)
            j_split_image = (j_split_image + 1)
        if j_split_image % num_of_splits == 0:
            i_original_image = (i_original_image + 1)

    downsize()

# Downsize the split images (better performance in yolov5)
def downsize():

    print("Downsizing split images...")

    # Make a folder (if one doesn't exist)
    if not os.path.exists("./splits_resized"):
        os.mkdir("./splits_resized")

    # Remove all files in the folder (if they exist)
    directory = 'splits_resized'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        os.remove(f)

    # Access global vars
    global split_list, new_annotation_list

    # Specifying a single scale factor
    scale_down_factor = 1/3

    # List to store the new resized splits
    resized_split_list = []

    # For loop to resize each split in the split_list
    for i in range(len(split_list)):

        # Grab the split and store it
        split_img = cv2.imread(split_list[i]['file_name'])

        # Scale that split down
        scaled_down_split = cv2.resize(split_img, None, fx=scale_down_factor, fy=scale_down_factor, interpolation=cv2.INTER_LINEAR)

        # Filename of each resized split
        file_name = './splits_resized/' + str(hash(str(i // num_of_splits) + '_' + str(i % num_of_splits))) + '.jpg'

        # Saving the split
        cv2.imwrite(file_name, scaled_down_split)

        # Write resized image
        write_new_image(split_list[i]['id'], split_list[i]['license'], file_name, str(int(split_list[i]['height'] * scale_down_factor)), str(int(split_list[i]['width'] * scale_down_factor)), split_list[i]['date_captured'], resized_split_list)


    # Resize all of the annotations in the new_annotation_list
    resized_new_annotation_list = []
    for i in range(len(new_annotation_list)):

        # Calculate new bbox and area
        downsized_bbox = [int(new_annotation_list[i]['bbox'][0] * scale_down_factor), int(new_annotation_list[i]['bbox'][1] * scale_down_factor), int(new_annotation_list[i]['bbox'][2] * scale_down_factor), int(new_annotation_list[i]['bbox'][3] * scale_down_factor)]
        downsized_area = int(downsized_bbox[2] * downsized_bbox[3])

        # Write new annotation
        write_new_annotation(new_annotation_list[i]['id'], new_annotation_list[i]['image_id'], new_annotation_list[i]['category_id'], downsized_bbox, downsized_area, new_annotation_list[i]['segmentation'], new_annotation_list[i]['iscrowd'], resized_new_annotation_list)

    # Write everything to JSON
    write_to_json(resized_split_list, resized_new_annotation_list)

# Convert coco file format to yolov5 file format
def convert_json2yolo():

    print("Converting to yolo...")

    # Call the other script to convert the coco to yolo (and prepare the yolo file for use)
    coco_to_yolo.ConvertCOCOToYOLO("./splits_resized", "./splits_resized/_new_annotations.coco.json", destination_dir, "./presplit" + dataset_dir + "/train/").convert()


# Clear the dirs created during Cocosplit
def clear_dirs():

    # Delete splits folder
    shutil.rmtree('./splits/')

    # Delete splits_resized folder
    shutil.rmtree('./splits_resized/')


## COCO SPLIT HELPERS ##
# Method to write new image in coco format
def write_new_image(id, license, file_name, height, width, date_captured, split_list):
    new_image = {
        "id": id,
        "license": license,
        "file_name": file_name,
        "height": height,
        "width": width,
        "date_captured": date_captured
    }

    split_list.append(new_image)

# Method to write new annotation in coco format
def write_new_annotation(id, image_id, category_id, bbox, area, segmentation, iscrowd, new_annotation_list):
    new_annotation = {
        "id" : id,
        "image_id" : image_id,
        "category_id" : category_id,
        "bbox" : bbox,
        "area" : area,
        "segmentation" : segmentation,
        "iscrowd" : iscrowd
    }

    new_annotation_list.append(new_annotation)

# Separate method for appending to split_list, and writing
def write_to_json(split_list, new_annotation_list):
    for i in range(len(split_list)):
        new_coco_data['images'].append(split_list[i])

    for i in range(len(new_annotation_list)):
        new_coco_data['annotations'].append(new_annotation_list[i])


################################################
# Clear out unused dirs
clear_dirs()

# Starts the program essentially
#sort_annotations()

# Save everything to the new JSON file
with open('./splits_resized/_new_annotations.coco.json', 'w') as file2:
    json.dump(new_coco_data, file2)

# Converts the coco json to yolo format for training
convert_json2yolo()

# Clear out unused dirs
clear_dirs()
################################################