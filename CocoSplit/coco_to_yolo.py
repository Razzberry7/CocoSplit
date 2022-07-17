import os
import shutil
import cv2
import json

class ConvertCOCOToYOLO:

    """
    Takes in the path to COCO annotations and outputs YOLO annotations in multiple .txt files.
    COCO annotation are to be JSON format as follows:

        "annotations":{
            "area":2304645,
            "id":1,
            "image_id":10,
            "category_id":4,
            "bbox":[
                0::704
                1:620
                2:1401
                3:1645
            ]
        }
        
    """

    def __init__(self, img_folder, json_path, zip_filename, original_img_path):
        self.img_folder = img_folder
        self.json_path = json_path
        self.zip_filename = zip_filename
        self.original_img_path = original_img_path
        

    def get_img_shape(self, img_path):
        img = cv2.imread(img_path)
        try:
            return img.shape
        except AttributeError:
            print('error!', img_path)
            return (None, None, None)

    def convert_labels(self, img_path, x1, y1, x2, y2):
        """
        Definition: Parses label files to extract label and bounding box
        coordinates. Converts (x1, y1, x1, y2) KITTI format to
        (x, y, width, height) normalized YOLO format.
        """

        def sorting(l1, l2):
            if l1 > l2:
                lmax, lmin = l1, l2
                return lmax, lmin
            else:
                lmax, lmin = l2, l1
                return lmax, lmin

        size = self.get_img_shape(img_path)
        xmax, xmin = sorting(x1, x2)
        ymax, ymin = sorting(y1, y2)
        dw = 1./size[1]
        dh = 1./size[0]
        x = (xmin + xmax)/2.0
        y = (ymin + ymax)/2.0
        w = xmax - xmin
        h = ymax - ymin
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x,y,w,h)

    def convert(self,annotation_key='annotations',img_id='image_id',cat_id='category_id',bbox='bbox'):
        # Enter directory to read JSON file
        data = json.load(open(self.json_path))
        
        check_set = set()

        ### Create new directory based off of original zip name
        parent_dir = "./" + self.zip_filename
        # Create a new folder (if one doesn't exist already)
        if not os.path.exists(parent_dir):
            os.mkdir(parent_dir)
        # Delete all existing files in that folder (if any exist)
        else:
            shutil.rmtree(parent_dir)
            os.mkdir(parent_dir)

        ### Create a directory to hold the original images
        parent_image_dir = parent_dir + "/original_images/"
        # Create a new folder (if one doesn't exist already)
        if not os.path.exists(parent_image_dir):
            shutil.copytree(self.original_img_path, parent_image_dir)
        else:
            shutil.rmtree(parent_image_dir)
            shutil.copytree(self.original_img_path, parent_image_dir)
        # Remove the json file
        files = os.listdir(parent_image_dir)
        for file in files:
            if file.endswith(".json"):
                f = os.path.join(parent_image_dir, file)
                os.remove(f)


        ### Create a train directory
        train_dir = parent_dir + "/train/"
        # Create a new folder (if one doesn't exist already)
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        ### Create an image directory
        train_image_dir = train_dir + "images/"
        # Create a new folder (if one doesn't exist already)
        if not os.path.exists(train_image_dir):
            shutil.copytree(self.img_folder, train_image_dir)
        else:
            shutil.rmtree(train_image_dir)
            shutil.copytree(self.img_folder, train_image_dir)
        # Remove the json file
        files = os.listdir(train_image_dir)
        for file in files:
            if file.endswith(".json"):
                f = os.path.join(train_image_dir, file)
                os.remove(f)

        ### Create a labels directory
        train_label_dir = train_dir + "labels/"
        # Create a new folder (if one doesn't exist already)
        if not os.path.exists(train_label_dir):
            os.mkdir(train_label_dir)

        ### Create a data.yaml file
        with open(parent_dir + '/data.yaml', 'w') as f:
            f.write('train: ../data/weights/' + self.zip_filename + '/train/images\n')
            f.write('val: ../data/weights/' + self.zip_filename + '/test/images\n')
            f.write('\n')
            f.write('nc: 3\n')
            f.write("names: ['berries', 'blue', 'green']")


        # Retrieve data
        for i in range(len(data[annotation_key])):

            # Get required data
            image_id = f'{data[annotation_key][i][img_id]}'
            category_id = f'{data[annotation_key][i][cat_id]}'
            boundbox = data[annotation_key][i][bbox]

            # Retrieve image.
            if self.img_folder == None:
                image_path = f'{image_id}.jpg'
            else:
                image_path = f'{self.img_folder}/{image_id}.jpg'


            # Convert the data
            kitti_bbox = [boundbox[0], boundbox[1], boundbox[2] + boundbox[0], boundbox[3] + boundbox[1]]
            yolo_bbox = self.convert_labels(image_path, kitti_bbox[0], kitti_bbox[1], kitti_bbox[2], kitti_bbox[3])



            # Prepare for export
            filename = f'{train_label_dir}{image_id}.txt'
            # Save to run folder that can be renamed
            content =f"{category_id} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}"

            # Export 
            if image_id in check_set:
                # Append to existing file as there can be more than one label in each image
                file = open(filename, "a")
                file.write("\n")
                file.write(content)
                file.close()

            elif image_id not in check_set:
                check_set.add(image_id)
                # Write files
                file = open(filename, "w")
                file.write(content)
                file.close()




# To run in as a class
if __name__ == "__main__":
    ConvertCOCOToYOLO(img_path='./splits_resized',json_path='./splits_resized/_new_annotations.coco.json').convert()
