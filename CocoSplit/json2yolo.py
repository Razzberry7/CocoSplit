import json
import glob

from PIL.Image import Image

def convert():
    data_path = glob.glob("./splits_resized/*.json")
    img_data_path = glob.glob("./splits_resized/*.jpg ")
    cate_list = ["berry", "blue", "green"]
    box_list = ["x1", "x2", "y1", "y2"]
    input_shape = [640, 640]
    img_output = "./yoloConvert/"

    with open("berry_classes.txt", "w") as f:
        f.write('\n'.join(cate_list))

    list_file = open("new_test.txt", "w")


    def class_encord(class_name):
        cate_id = {"berry": 0, "blue": 1, "green": 2}
        return cate_id[class_name]


    def img_box_resize(img_path, input_shape, label_box):
        img = Image.open(img_path)
        iw, ih = img.size
        w, h = input_shape
        scale = min(w / iw, h / ih)
        nw = int(scale * iw)
        nh = int(scale * ih)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        img_data = 0
        image = img.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        box_lx = label_box[0]
        box_ly = label_box[1]
        box_rx = label_box[2]
        box_ry = label_box[3]

        new_box_lx = int(box_lx * scale) + int(dx)
        new_box_ly = int(box_ly * scale) + int(dy)
        new_box_rx = int(box_rx * scale) - int(dx)
        new_box_ry = int(box_ry * scale) - int(dy)
        box_list = []
        box_list.append(new_box_lx)
        box_list.append(new_box_ly)
        box_list.append(new_box_rx)
        box_list.append(new_box_ry)
        return new_image, box_list


    def convert(data_file, list_file, input_img_path, output_img_path):
        with open(data_file) as j_f:
            load_json = json.load(j_f)
            datas = load_json["images"]
            for data in datas:
                box = data["box2d"]
                json_box = []
                new_img, box_list = img_box_resize(img_path, input_shape, box)
                for i in range(4):
                    json_box.append(box_list[i])
                category = data["category"]
                category_num = class_encord(category)
                list_file.write(" " + ",".join([str(a) for a in json_box]) + "," + str(category_num))
                new_img.save(output_img_path + "/" + input_img_path)


    for data_file in data_path:
        for img_path in img_data_path:
            #data = data_file
            list_file.write(img_output + "/" + img_path)
            convert(data_file, list_file, img_path, img_output)
            list_file.write("\n")

    list_file.close()