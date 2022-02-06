import cv2
import json


scale_factor = 1

# Original annotation changing
file = open('_annotations.coco.json')
data = json.load(file)
file.close()

data2 = \
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

def slice_images():
    #Indices
    i = 0

    # Nested while loop that splits image by columns for each row
    while i < 10:

        # Import image
        original_img = cv2.imread(str(i) + '.jpg')

        # Pad the image to make it divisible by 640 in width and height
        #edited_image = cv2.copyMakeBorder(original_img, 0, 192, 0, 288, cv2.BORDER_CONSTANT)
        edited_image = cv2.resize(original_img, (5760, 3840), interpolation = cv2.INTER_AREA)
        global scale_factor
        scale_factor = 1.0526315789473684210526315789474


        # Grabs height and width of the padded_image (change for different picture)
        h, w, channels = edited_image.shape

        # Number of columns/rows needed for the image
        w_iterations = (w / 640)
        h_iterations = (h / 640)

        # Table by row of all columns of images
        height_table = []

        # List of columns
        width_sections = []

        j = 0
        k = 0
        while j < h_iterations:
            while k < w_iterations:
                #Split code (change padded_image for different picture)
                width_sections.append(edited_image[(j * 640):((j + 1) * 640), (k * 640):((k + 1) * 640)])
                #Filename of each split
                file_name = 'section' + str(i) + '_' + str(j) + '_' + str(k) + '.jpg'
                #Saving the split
                cv2.imwrite(file_name, width_sections[k])

                ####Writing the images to the JSON file

                #Create the id var
                #id = ((i * 53) ) + (j * 9) + (k)
                id = hash(str(i) + str(j) + str(k))


                #Create the data_captured var
                images = data['images']
                date_captured = images[i]['date_captured']

                write_new_image(id, 1, file_name, 640, 640, date_captured)

                k = (k + 1)
            height_table.append(list(width_sections))
            width_sections.clear()
            k = 0
            j = (j + 1)

        i = (i + 1)

def write_new_image(id, license, file_name, height, width, date_captured):
    new_image = {
        "id": id,
        "license": license,
        "file_name": file_name,
        "height": height,
        "width": width,
        "date_captured": date_captured
    }

    data2['images'].append(new_image)

# cv2.imwrite() function will save the image into your pc
##cv2.imwrite('paddedimage.jpg', padded_image)
##cv2.waitKey(0)

def adjust_annotations():
    annotations = data['annotations']

    y=0
    while y < len(annotations):
        #print(annotations[y])

        old_annot = annotations[y]['bbox']
        print(annotations[y]['bbox'])

        old_x = int(old_annot[0]*scale_factor)
        old_y = int(old_annot[1]*scale_factor)
        old_width = int(old_annot[2]*scale_factor)
        old_height = int(old_annot[3]*scale_factor)

        #New id
        id1 = len(data2['annotations'])

        #New image_id
        image_id1 = hash(str((annotations[y]['image_id'])) + str(old_y//640) + str(old_x//640))

        #New category_id
        category_id = annotations[y]['category_id']

        # New origin for original annotation
        new_x1 = old_x%640
        new_y1 = old_y%640

        #New width
        if (640 - new_x1 > old_width):
            new_width1 = old_width
        else:
            new_width1 = 640 - new_x1
        #New height
        if (640 - new_y1 > old_height):
            new_height1 = old_height
        else:
            new_height1 = 640 - new_y1

        #New bbox list
        new_bbox1 = [new_x1, new_y1, new_width1, new_height1]

        #New area
        new_area1 = new_width1 * new_height1

        #Write to JSON file
        write_new_annotation(id1, image_id1, category_id, new_bbox1, new_area1, [], 0)

        #print('bbox1:')
        #print(bbox1)

        # Does it exceed the size of its picture to the east?
        if new_x1 + old_width > 640:
            # New id
            id2 = len(data2['annotations'])

            # New image_id (+1 because it's to the east)
            image_id2 = hash(str((annotations[y]['image_id'])) + str(old_y//640) + str(old_x//640))

            #New origin for new east annotation
            new_x2 = 0
            new_y2 = new_y1

            #New width
            new_width2 = (new_x1 + old_width) % 640

            # New height
            new_height2 = new_height1

            # New bbox list
            new_bbox2 = [new_x2, new_y2, new_width2, new_height2]

            # New area
            new_area2 = new_width2 * new_height2

            #print('bbox2:')
            #print(bbox2)
            # Write to JSON file using last ID + 1
            write_new_annotation(id2, image_id2, category_id, new_bbox2, new_area2, [], 0)

        # Does it exceed the size of its picture to the south?
        if new_y1 + old_height > 640:
            # New id
            id3 = len(data2['annotations'])

            # New image_id (+9 because it's to the south)
            image_id3 = hash(str((annotations[y]['image_id'])) + str(old_y//640) + str(old_x//640))

            #New origin for new south annotation
            new_x3 = new_x1
            new_y3 = 0

            # New width
            new_width3 = new_width1

            #New height
            new_height3 = (new_y1 + old_height) % 640

            # New bbox list
            new_bbox3 = [new_x3, new_y3, new_width3, new_height3]

            # New area
            new_area3 = new_width3 * new_height3

            #print('bbox3:')
            #print(bbox3)
            # Write to JSON file using last ID + 1
            write_new_annotation(id3, image_id3, category_id, new_bbox3, new_area3, [], 0)

        # Does it exceed the size of its picture to both east and south?
        if new_width1 != old_width and new_height1 != old_height:
            # New id
            id4 = len(data2['annotations'])

            # New image_id (+53 because it's to the South-east)
            image_id4 = hash(str((annotations[y]['image_id'])) + str(old_y//640) + str(old_x//640))

            #New origin for new south-east annotation
            new_x4 = 0
            new_y4 = 0

            #New width
            new_width4 = new_width2

            # New height
            new_height4 = new_height3

            # New bbox list
            new_bbox4 = [new_x4, new_y4, new_width4, new_height4]

            # New area
            new_area4 = new_width4 * new_height4

            #print('bbox4:')
            #print(bbox4)
            # Write to JSON file using last ID + 1
            write_new_annotation(id4, image_id4, category_id, new_bbox4, new_area4, [], 0)

        y = y + 1



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

    data2['annotations'].append(new_annotation)


def try_new_annotation():
    id = 9
    image_id = 9
    category_id = 9
    bbox = [
        9,
        9,
        9,
        9
    ]
    area = 9
    segmentation = []
    iscrowd = 9
    write_new_annotation(id, image_id, category_id, bbox, area, segmentation, iscrowd)

#try_new_annotation()
slice_images()
adjust_annotations()

with open('_annotations.coco_copy.json', 'w') as file2:
    json.dump(data2, file2)