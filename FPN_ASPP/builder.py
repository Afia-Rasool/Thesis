import numpy as np
from keras.layers import Conv2D, ZeroPadding2D, DepthwiseConv2D
from keras.layers import Concatenate
from keras.layers import Activation
from keras.models import Model
from ..common import ResizeImage, Conv2DBlock
from ..common import extract_outputs, to_tuple
from keras.layers import Dropout, BatchNormalization
from keras.layers import Input
from keras.layers import Lambda
import tensorflow as tf
from keras.layers import GlobalAveragePooling2D
from keras import backend as K
from keras.layers import Add


def pyramid_block(pyramid_filters=256, segmentation_filters=128, upsample_rate=2,
                  use_batchnorm=False, stage=0):
    def layer(c, m=None):
        x = Conv2DBlock(pyramid_filters, (1, 1),
                        padding='same',
                        use_batchnorm=use_batchnorm,
                        name='pyramid_stage_{}'.format(stage))(c)
        if m is not None:
            up = ResizeImage(to_tuple(upsample_rate))(m)
            x = Add()([x, up])
        # segmentation head
        p = Conv2DBlock(segmentation_filters, (3, 3),
                        padding='same',
                        use_batchnorm=use_batchnorm,
                        name='segm1_stage_{}'.format(stage))(x)
        p = Conv2DBlock(segmentation_filters, (3, 3),
                        padding='same',
                        use_batchnorm=use_batchnorm,
                        name='segm2_stage_{}'.format(stage))(p)
        m = x
        return m, p
    return layer



def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'
    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x



def build_fpn_ASPP(backbone,
              fpn_layers,
              input_shape=(None, None, 3),
              classes=21,
              activation='softmax',
              upsample_rates=(2, 2, 2),
              pyramid_filters=256,
              segmentation_filters=128,
              use_batchnorm=False,
              dropout=None,
              interpolation='bilinear'):
    if len(upsample_rates) != len(fpn_layers):
        raise ValueError('Number of intermediate feature maps and upsample steps should match')
    outputs = extract_outputs(backbone, fpn_layers, include_top=True)
    upsample_rates = [1] + list(upsample_rates)
    # top - down path, build pyramid
    m = None
    pyramid = []
    for i, c in enumerate(outputs):
        m, p = pyramid_block(pyramid_filters=pyramid_filters,
                             segmentation_filters=segmentation_filters,
                             upsample_rate=upsample_rates[i],
                             use_batchnorm=use_batchnorm,
                             stage=i)(c, m)
        pyramid.append(p)
    # upsample and concatenate all pyramid layer
    upsampled_pyramid = []
    for i, p in enumerate(pyramid[::-1]):
        if upsample_rates[i] > 1:
            upsample_rate = to_tuple(np.prod(upsample_rates[:i + 1]))
            p = ResizeImage(upsample_rate, interpolation=interpolation)(p)
        upsampled_pyramid.append(p)
    x = Concatenate()(upsampled_pyramid)
    img_input = Input(shape=input_shape)
    atrous_rates = (6, 12, 18)
    # Image Feature branch
    b4 = GlobalAveragePooling2D()(x)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3],
                                                    method='bilinear', align_corners=True))(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)
    # rate = 6 (12)
    b1 = SepConv_BN(x, 256, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # rate = 12 (24)
    b2 = SepConv_BN(x, 256, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # rate = 18 (36)
    b3 = SepConv_BN(x, 256, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)
    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])

    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    # DeepLab v.3+ decoder
    size_before2 = tf.keras.backend.int_shape(x)
    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                    size_before2[1:3] * tf.constant(4),
                                                    method='bilinear', align_corners=True))(x)
    dec_skip1 = Conv2D(48, (1, 1), padding='same',
                       use_bias=False, name='feature_projection0')(x)
    dec_skip1 = BatchNormalization(
        name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation('relu')(dec_skip1)
    x = Concatenate()([x, dec_skip1])
    x = SepConv_BN(x, 256, 'decoder_conv0',
                   depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256, 'decoder_conv1',
                   depth_activation=True, epsilon=1e-5)
    last_layer_name = 'custom_logits_semantic'
    x = Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
    size_before3 = tf.keras.backend.int_shape(img_input)
    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                    size_before3[1:3],
                                                    method='bilinear', align_corners=True))(x)
    if activation in {'softmax', 'sigmoid'}:
        x = Activation(activation)(x)
    model = Model(backbone.input, x)
    return model




