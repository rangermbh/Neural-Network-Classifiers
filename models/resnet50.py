"""ResNet50 model for Keras, just copy this code, and try to get a better understanding of this model.

# Reference:

- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
"""

from __future__ import print_function

import numpy as np
import warnings

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
import keras.backend as K
from keras.layers import BatchNormalization
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D

from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.models import Model
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.preprocessing import image

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    Blocks with no conv layer on the shortcut
    :param input_tensor: input_tensor
    :param kernel_size: the height and width of the 2D convolution window of middle conv layer at main path
    :param filters: list of integers, the filters of 3 conv layer at main path
    :param stage: integer, current stage label, used for generating layer names
    :param block: 'a', 'b'......, current block label, used for generating layer names
    :return: output tensor for the block
    """

    filter1, filter2, filter3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters=filter1, kernel_size=(1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation(activation='relu')(x)

    # padding = 'same'
    x = Conv2D(filters=filter2, kernel_size=kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation(activation='relu')(x)

    x = Conv2D(filters=filter3, kernel_size=(1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    Blocks with conv layer on the shortcut
    :param input_tensor:  input_tensor
    :param kernel_size:  receptive field size(height and width of filter) of the middle conv layer
    :param filters:  list of integers, the filters of 3 conv layer at main path
    :param stage:  integer, current stage label, used for generating layer names
    :param block:  'a', 'b'......, current block label, used for generating layer names
    :param strides:  stride of conv layer in shortcut
    :return: output tensor of this block
    """

    filter1, filter2, filter3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters=filter1, kernel_size=(1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation(activation='relu')(x)

    # padding='same' ?
    x = Conv2D(filters=filter2, kernel_size=kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation(activation='relu')(x)

    x = Conv2D(filters=filter3, kernel_size=(1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters=filter3, kernel_size=(1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def Resnet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the ResNet50 architecture.

       Optionally loads weights pre-trained on ImageNet.
       Note that the data format convention used by the model is
       the one specified in your Keras config at `~/.keras/keras.json`.

       # Arguments
           include_top: whether to include the fully-connected
               layer at the top of the network.
           weights: one of `None` (random initialization),
                 'imagenet' (pre-training on ImageNet),
                 or the path to the weights file to be loaded.
           input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
               to use as image input for the model.
           input_shape: optional shape tuple, only to be specified
               if `include_top` is False (otherwise the input shape
               has to be `(224, 224, 3)` (with `channels_last` data format)
               or `(3, 224, 224)` (with `channels_first` data format).
               It should have exactly 3 inputs channels,
               and width and height should be no smaller than 197.
               E.g. `(200, 200, 3)` would be one valid value.
           pooling: Optional pooling mode for feature extraction
               when `include_top` is `False`.
               - `None` means that the output of the model will be
                   the 4D tensor output of the
                   last convolutional layer.
               - `avg` means that global average pooling
                   will be applied to the output of the
                   last convolutional layer, and thus
                   the output of the model will be a 2D tensor.
               - `max` means that global max pooling will
                   be applied.
           classes: optional number of classes to classify images
               into, only to be specified if `include_top` is True, and
               if no `weights` argument is specified.

       # Returns
           A Keras model instance.

       # Raises
           ValueError: in case of invalid argument for `weights`,
               or invalid input shape.
       """

    if weights not in {'imagenet', None}:
        raise ValueError(
            "The 'weight' argumemt should be either 'None (random initialization)"
            " or 'imagenet'(pre-trained on ImageNet) ")
    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using "weights" as imagenet with "include_top" as true, classes should be 1000  ')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = conv_block(input_tensor=x, kernel_size=3, filters=[64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(input_tensor=x, kernel_size=3, filters=[64, 64, 256], stage=2, block='b')
    x = identity_block(input_tensor=x, kernel_size=3, filters=[64, 64, 256], stage=2, block='c')

    x = conv_block(input_tensor=x, kernel_size=3, filters=[128, 128, 512], stage=3, block='a')
    x = identity_block(input_tensor=x, kernel_size=3, filters=[128, 128, 512], stage=3, block='b')
    x = identity_block(input_tensor=x, kernel_size=3, filters=[128, 128, 512], stage=3, block='c')
    x = identity_block(input_tensor=x, kernel_size=3, filters=[128, 128, 512], stage=3, block='d')

    x = conv_block(input_tensor=x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='a')
    x = identity_block(input_tensor=x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='b')
    x = identity_block(input_tensor=x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='c')
    x = identity_block(input_tensor=x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='d')
    x = identity_block(input_tensor=x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='e')
    x = identity_block(input_tensor=x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='f')

    x = conv_block(input_tensor=x, kernel_size=3, filters=[512, 512, 2048], stage=5, block='a')
    x = identity_block(input_tensor=x, kernel_size=3, filters=[512, 512, 2048], stage=5, block='b')
    x = identity_block(input_tensor=x, kernel_size=3, filters=[512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D(pool_size=(7, 7), name="avg_pool")(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # create model

    model = Model(inputs, x, name='resnet50')

    # load weight

    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)

        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channel_first')

            if K.backend() == 'tensorflow':
                warnings.warn('you are using the tensorflow backend, yet you are using the theano image'
                              'data format convention'
                              '("image_data_format = channel_first)"'
                              'for best performance, set'
                              'image_data_format=channel_last in your keras config'
                              'at ~/.keras/keras.json')
    return model


if __name__ == '__main__':
    model = Resnet50(include_top=True, weights='imagenet')

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    #  Insert a new axis that will appear at the `axis` position in the expanded
    #     array shape.
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape: ', x.shape)

    preds = model.predict(x)
    print("get_weights()", model.get_config())

    print('Predicted: ', decode_predictions(preds))