import skimage.io
import numpy as np
import scipy.misc
import skimage.color
import chardet
import random


class Dataset(object):
    def __init__(self):
        self._image_ids = []
        self.image_info = []
        self.class_info = []

    @property
    def image_ids(self):
        return self._image_ids

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info['id'] == class_id:
                return

        # Add class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def to_string(self):
        print("\nDataset")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))

    def load_image(self, image_id):
        """
        Load a specified image and return a [H,W,3] Numpy array
        """
        path = self.image_info[image_id]['path']
        try:
            image = skimage.io.imread(path + '.jpg')
        except:
            image = skimage.io.imread(path + '.png')

        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

    def prepare(self):
        """
        prepare dataset for use

        """

        def clean_name(name):
            return ",".join(name.split(",")[:1])

        # Build or rebuild everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)

        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        for source in self.sources:
            self.source_class_ids[source] = []
            for i, info in enumerate(self.class_info):
                self.source_class_ids[source].append(i)


def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = scipy.misc.imresize(
            image, (round(h * scale), round(w * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding


def encode_file(file, encoding='utf-8'):
    """
    Encode file to specific encoding
    """
    with open(file, 'rb+') as fr:
        content = fr.read()
        current_encoding = chardet.detect(content)['encoding']
        content = content.decode(current_encoding).encode(encoding)
        # write back
        fr.seek(0)
        fr.write(content)


if __name__ == '__main__':
    dataset = Dataset()
    dataset.to_string()
    dataset.add_class("huaxi", 1, 'optimum')
    dataset.add_image("huaxi", "elephant.jpg", "/Users/moubinhao/PycharmProjects/tensorflow_keras/elephant.jpg")
    dataset.to_string()
    dataset.load_image(0)
