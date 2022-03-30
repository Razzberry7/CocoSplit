import random

import cv2
import json
import os
import zipfile

num_of_splits = 10
split_list = []
new_annotation_list = []

image_annotation_lists = []
split_origins = []



w_iterations = 0
h_iterations = 0
dimensionX = 1024
dimensionY = 640


# Unzip roboflow zip

path = "./presplit/"
files = os.listdir(path)
for file in files:
    if file.endswith(".zip"):
        filePath = path + "/" + file
        zip_file = zipfile.ZipFile(filePath)
        for names in zip_file.namelist():
            zip_file.extract(names, path)
        zip_file.close()

#os.remove("./presplit/*.zip")

# Original annotation changing
file = open('./presplit/train/_annotations.coco.json')
old_coco_data = json.load(file)
file.close()

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
                "name": "berry",
                "supercategory": "none"
            },
            {
                "id": 1,
                "name": "blue",
                "supercategory": "berry"
            },
            {
                "id": 2,
                "name": "green",
                "supercategory": "berry"
            }
        ],
        "images": [],
        "annotations": []
    }

def sort_annotations():

    # dictionary of annotations from original coco file
    annotations = old_coco_data['annotations']

    # sort annotations by image_id and store it
    sorted_annotations = sorted(annotations, key=lambda x: x['image_id'])

    # Call next step and pass in sorted_annotations list to be separated
    split_sorted_annotations(sorted_annotations)

def split_sorted_annotations(sorted_annotations):

    # dictionary of images from original coco file
    images = old_coco_data['images']

    global image_annotation_lists
    for n in range(len(images)):
        image_annotation_lists.append([])

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

def random_split():

    # dictionary of images from original coco file
    images = old_coco_data['images']

    os.mkdir("./splits")
    directory = 'splits'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        os.remove(f)

    # Indices
    i = 0

    # split origins (x,y)
    global split_origins

    # Nested while loop that splits images
    while i < len(os.listdir('./presplit/train')) - 1:
        # Import image
        original_img = cv2.imread('./presplit/train/' + images[i]["file_name"])

        # List of splits for each picture
        splits = []

        j = 0
        while j < num_of_splits:
            # Split code (change padded_image for different picture)

            #random_height = random.randint(500, 800)
            #random_width = random.randint(900, 1200)
            random_height = int(images[i]['height'] / 2)
            random_width = int(images[i]['width'] / 2)
            random_y = random.randint(0, (images[i]['height'] - random_height))
            random_x = random.randint(0, (images[i]['width'] - random_width))

            splits.append(original_img[random_y:(random_y + random_height), random_x:(random_x + random_width)])
            split_origins.append([random_x, random_y])

            # Filename of each split
            file_name = './splits/' + str(i) + '_' + str(j) + '_' + str(random_width) + 'x' + str(random_height) + '.jpg'
            # Saving the split
            cv2.imwrite(file_name, splits[j])

            ####Writing the images to the JSON file

            # Create the id var
            id = hash(str(i) + "_" +  str(j))

            # Create the data_captured var
            images = old_coco_data['images']
            date_captured = images[i]['date_captured']

            write_new_image(id, 1, file_name, random_height, random_width, date_captured)

            j = (j + 1)

        i = (i + 1)
    adjust_random_annotations()

def adjust_random_annotations():
    global image_annotation_lists, split_origins

    original_images = old_coco_data['images']
    split_images = split_list
    split_annotations = new_annotation_list

    # original_image index for grabbing the annotations list corresponding to the original image
    original_image = 0

    i_original_image = 0
    j_split_image = 0
    k_blueberry = 0

    # for loop for the number of original pictures
    for i_original_image in range(len(original_images)):
        annotation_list = image_annotation_lists[i_original_image]
        # for loop for the number of split images
        for j_split_image in range(num_of_splits):
            curr_split = split_images[i_original_image * num_of_splits + j_split_image]
            curr_split_origin = split_origins[i_original_image * num_of_splits + j_split_image]

            # for loop for the number of blueberries
            for k_blueberry in range(len(image_annotation_lists[i_original_image])):
                old_annot = annotation_list[k_blueberry]['bbox']

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

                # Write to JSON file
                write_new_annotation(id1, image_id1, category_id, new_bbox1, new_area1, [], 0)

                k_blueberry = (k_blueberry + 1)
            j_split_image = (j_split_image + 1)
        if j_split_image % num_of_splits == 0:
            i_original_image = (i_original_image + 1)

    downsize()

# Downsize the split images (better performance in yolov5)
def downsize():

    os.mkdir("./splits_resized")
    directory = 'splits_resized'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        os.remove(f)

    global split_list, new_annotation_list

    # Scaling Down the image 1/3 times specifying a single scale factor.
    scale_down = 1/3

    resized_split_list = []
    for i in range(len(split_list)):
        split_img = cv2.imread(split_list[i]['file_name'])
        scaled_f_down = cv2.resize(split_img, None, fx=scale_down, fy=scale_down, interpolation=cv2.INTER_LINEAR)

        # Filename of each split
        file_name = './splits_resized/' + str(i) + '--' + str(int(split_list[i]['width'] * scale_down)) + 'x' + str(int(split_list[i]['height'] * scale_down)) + '.jpg'

        # Saving the split
        cv2.imwrite(file_name, scaled_f_down)

        downsized_new_image = {
            "id": split_list[i]['id'],
            "license": split_list[i]['license'],
            "file_name": file_name,
            "height": str(int(split_list[i]['height'] * scale_down)),
            "width": str(int(split_list[i]['width'] * scale_down)),
            "date_captured": split_list[i]['date_captured']
        }

        resized_split_list.append(downsized_new_image)


    resized_new_annotation_list = []
    for i in range(len(new_annotation_list)):

        downsized_bbox = [int(new_annotation_list[i]['bbox'][0] * scale_down), int(new_annotation_list[i]['bbox'][1] * scale_down), int(new_annotation_list[i]['bbox'][2] * scale_down), int(new_annotation_list[i]['bbox'][3] * scale_down)]
        downsized_area = int(downsized_bbox[2] * downsized_bbox[3])

        downsized_new_annotation = {
            "id": new_annotation_list[i]['id'],
            "image_id": new_annotation_list[i]['image_id'],
            "category_id": new_annotation_list[i]['category_id'],
            "bbox": downsized_bbox,
            "area": downsized_area,
            "segmentation": new_annotation_list[i]['segmentation'],
            "iscrowd": new_annotation_list[i]['iscrowd']
        }
        resized_new_annotation_list.append(downsized_new_annotation)

    write_to_json(resized_split_list, resized_new_annotation_list)



## COCO SPLIT HELPERS
def write_new_image(id, license, file_name, height, width, date_captured):
    new_image = {
        "id": id,
        "license": license,
        "file_name": file_name,
        "height": height,
        "width": width,
        "date_captured": date_captured
    }

    global split_list
    split_list.append(new_image)

    #new_coco_data['images'].append(new_image)


def write_new_annotation(id, image_id, category_id, bbox, area, segmentation, iscrowd):
    new_annotation = {
        "id" : id,
        "image_id" : image_id,
        "category_id" : category_id,
        "bbox" : bbox,
        "area" : area,
        "segmentation" : segmentation,
        "iscrowd" : iscrowd
    }

    global new_annotation_list
    new_annotation_list.append(new_annotation)

    #new_coco_data['annotations'].append(new_annotation)

# separate method for appending to split_list, and writing
def write_to_json(split_list, new_annotation_list):
    for i in range(len(split_list)):
        new_coco_data['images'].append(split_list[i])

    for i in range(len(new_annotation_list)):
        new_coco_data['annotations'].append(new_annotation_list[i])


sort_annotations()

with open('./splits_resized/_new_annotations.coco.json', 'w') as file2:
    json.dump(new_coco_data, file2)