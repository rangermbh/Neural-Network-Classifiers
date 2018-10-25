import os
import sys
import json
import numpy as np
import utils
import model as modellib
from config import Config

import visualize
from keras import utils  as k_utils

ROOT_DIR = os.getcwd()

sys.path.append(ROOT_DIR)


############################################################
#  Configurations
############################################################


class HuaxiConfig(Config):
    """Configuration for training on huaxi dataset
    Derives from the base Config class and overrides values specific
    to the huaxi dataset.
    """
    # Give the configuration a recognizable name
    NAME = "huaxi"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes
    NUM_CLASSES = 2  # hauxi has 2 classes

    # 在分类任务中没有用，等于类别数
    MAX_GT_INSTANCES = 2


class HuaxiDataset(utils.Dataset):
    def load_huaxi(self, data_dir, subset, format="json"):
        """Load a subset of the huaxi dataset.
             dataset_dir: Root directory of the dataset.
             subset: Subset to load: train or val
        """
        # Add classes.
        self.add_class("huaxi", 0, "malignant")
        self.add_class("huaxi", 1, "optimum")

        # Train, test or validation dataset

        assert subset in ["train", "val", "test"]
        json_dir = os.path.join(data_dir, "json", subset)
        img_dir = os.path.join(data_dir, "img")
        if format == "json":
            # read from json
            # when read data from json, file are store in this format:
            # 1. data_dir/img
            # 2. data_dir/json/train (or test)
            for file in os.listdir(json_dir):
                if file == '.DS_Store': continue
                # print(file)
                annotation = json.load(open(os.path.join(json_dir, file)))
                # print(annotation)
                # 由于json文件里的imagePath字段格式不统一，通过读取根目录的方式获取文件
                # print("imagePath = ", annotation['imagePath'])
                # print("suffix = ", suffix)
                file_name = file.split('.')[0]
                # print("file_name = ", file_name)
                image_path = os.path.join(img_dir, file_name)
                shapes_lable = [shape['label'] for shape in annotation['shapes']]
                shapes_lable_dict = {"opt": 1, "ma": 0}
                shapes_id = [shapes_lable_dict[a] for a in shapes_lable]
                # polygons = [p['points'] for p in annotation['shapes']]
                self.add_image(
                    "huaxi",
                    image_id=file_name,
                    path=image_path,
                    class_id=shapes_id,
                    polygons=None
                )
        else:
            # Directly read classified images, e.g /data_dir/opt/89887/89887.jpg
            for label in os.listdir(data_dir):
                if label == "opt":
                    shapes_id = 1
                elif label == "ma":
                    shapes_id = 0
                for image_id in os.listdir(os.path.join(data_dir, label)):
                    for images in os.listdir(os.path.join(data_dir, label, image_id)):
                        image_path = os.path.join(data_dir, label, image_id, images.split('.')[0])
                        self.add_image(
                            "huaxi",
                            image_id=images.split('.')[0],
                            path=image_path,
                            class_id=[shapes_id]
                        )

            # Directly read from file system, image are store in opt/1.jpg or ma/2.jpg

    def load_data(self, subset, config):
        """Loads image and store in array of current subset, if train, x stand for x_train

        # Returns
            Tuple of Numpy arrays: `(x_train, y_train) or (x_test, y_test)`.
        """
        num_sample = self.num_images
        x = np.empty((num_sample, 32, 32, 3), dtype='uint8')
        y = np.empty((num_sample,), dtype='uint8')

        for image_id in self.image_ids:
            image = self.load_image(image_id)
            image = utils.resize_image(image, min_dim=0, max_dim=32, padding=True)[0]
            x[image_id] = image
            # print(self.image_info[image_id]['class_id'])
            # 由于之前考虑了多区域，所以class_id是个数组，这里只取第一个
            # print(self.image_info[image_id]['class_id'])
            y[image_id] = self.image_info[image_id]['class_id'][0]

        y = np.reshape(y, (len(y), 1))
        return (x, y)


def test_format():
    # data_dir = "/Users/moubinhao/programStaff/huaxi_data/json"
    # for file in os.listdir(data_dir):
    #     # print(file)
    #     annotation = json.load(open(os.path.join(data_dir, file)))
    #     # print(annotation)
    #     suffix = annotation['imagePath'][-3:]
    #     print(suffix)
    #     print(annotation['imagePath'])
    #     shapes_lable = [shape['label'] for shape in annotation['shapes']]
    #     shapes_lable_dict = {"opt": 1, "ma": 0}
    #     shapes_id = [shapes_lable_dict[a] for a in shapes_lable]
    #     polygons = [p['points'] for p in annotation['shapes']]
    #     print(polygons)

    # visualize test
    # image_ids = np.random.choice(huaxi.image_ids, 4)
    # print(image_ids)
    # images_array = []
    # for image_id in image_ids:
    #     image = huaxi.load_image(image_id)
    #     images_array.append(image)
    # visualize.display_images(images_array)

    data_dir = "/Users/moubinhao/programStaff/huaxi_test_data"
    for label in os.listdir(data_dir):
        print(label)
        if label == '.DS_Store': continue
        if label == "opt":
            shapes_id = 1
        elif label == "ma":
            shapes_id = 0
        for image_id in os.listdir(os.path.join(data_dir, label)):
            for images in os.listdir(os.path.join(data_dir, label, image_id)):
                image_path = os.path.join(data_dir, label, image_id, images)
                print(image_path)


if __name__ == '__main__':
    # load train and test data
    train_data_dir = "/Users/moubinhao/programStaff/huaxi_data"
    test_data_dir = "/Users/moubinhao/programStaff/huaxi_test_data"

    huaxi_train = HuaxiDataset()
    huaxi_train.load_huaxi(train_data_dir, 'train', "json")
    huaxi_train.load_huaxi(test_data_dir, 'test', 'others')
    huaxi_train.prepare()
    huaxi_train.to_string()
    # x_train, y_train = huaxi_train.load_data('train')

    huaxi_test = HuaxiDataset()
    huaxi_test.load_huaxi(test_data_dir, 'test', "others")
    huaxi_test.prepare()
    huaxi_test.to_string()
    # x_test, y_test = huaxi_test.load_data('test')

    # Normalize data
    # x_train = x_train.astype('float32') / 255
    # x_test = x_test.astype('float32') / 255

    # if subtract pixel mean is enabled
    # subtract_pixel_mean = True
    # if subtract_pixel_mean:
    #     x_train_mean = np.mean(x_train, axis=0)
    #     x_train -= x_train_mean
    #
    # print('x_train shape', x_train.shape)
    # print('y_train shape', y_train.shape)
    # print('x_test shape', x_test.shape)
    # print('y_test shape', y_test.shape)
    # # convert class vector to binary class metrics
    # y_train = k_utils.to_categorical(y_train, 2)
    # y_test = k_utils.to_categorical(y_test, 2)

    # import models.cifar10_resnet as model
    #
    # model.train_model(x_train, y_train, x_test, y_test)
    # import test

    #
    # config = Config()
    # config.MAX_GT_INSTANCES = 1
    # config.display()
    # train_generator = test.data_generator(huaxi_train, config, batch_size=1, shuffle=False)
    # input = next(train_generator)
    # print(input, "input ....................")
    # print(next(train_generator))
    # print("num_images_epoch", huaxi_train.num_images)
    # for i in range(20):
    #     print("index ", i)
    #     next(train_generator)
    #
    # list = huaxi_train.image_info[0]['class_id']
    # list = np.array(list, dtype=np.int32)
    # print(type(list))
    # print(np.any(list > 0))

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    config = HuaxiConfig()
    # config.display()
    model = modellib.Resnet(mode='training', config=config, model_dir=MODEL_DIR)
    model.train(train_dataset=huaxi_train,
                val_dataset=huaxi_test,
                learning_rate=config.LEARNING_RATE,
                epochs=2,
                layers=None)
