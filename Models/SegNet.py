"""
@author: LiShiHang
@software: PyCharm
@file: SegNet.py
@time: 2018/12/18 14:58
"""
from keras import Model,layers
from keras.layers import Input,Conv2D,BatchNormalization,Activation,Reshape

from Models.utils import MaxUnpooling2D,MaxPoolingWithArgmax2D

def SegNet(nClasses, input_height, input_width):

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=( input_height, input_width,3))

    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x, mask_1 = MaxPoolingWithArgmax2D(name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x , mask_2 = MaxPoolingWithArgmax2D(name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x, mask_3 = MaxPoolingWithArgmax2D(name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)

    x, mask_4 = MaxPoolingWithArgmax2D(name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x, mask_5 = MaxPoolingWithArgmax2D(name='block5_pool')(x)

    Vgg_streamlined=Model(inputs=img_input,outputs=x)

    # 加载vgg16的预训练权重
    Vgg_streamlined.load_weights(r"E:\Code\PycharmProjects\keras-segmentation\data\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")

    # 解码层
    unpool_1 = MaxUnpooling2D()([x, mask_5])
    y = Conv2D(512, (3,3), padding="same")(unpool_1)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(512, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(512, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    unpool_2 = MaxUnpooling2D()([y, mask_4])
    y = Conv2D(512, (3, 3), padding="same")(unpool_2)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(512, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(256, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    unpool_3 = MaxUnpooling2D()([y, mask_3])
    y = Conv2D(256, (3, 3), padding="same")(unpool_3)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(256, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(128, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    unpool_4 = MaxUnpooling2D()([y, mask_2])
    y = Conv2D(128, (3, 3), padding="same")(unpool_4)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(64, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    unpool_5 = MaxUnpooling2D()([y, mask_1])
    y = Conv2D(64, (3, 3), padding="same")(unpool_5)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    y = Conv2D(nClasses, (1, 1), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    y = Reshape((-1, nClasses))(y)
    y = Activation("softmax")(y)

    model=Model(inputs=img_input,outputs=y)
    return model



if __name__ == '__main__':
    m = SegNet(15,320, 320)
    # print(m.get_weights()[2]) # 看看权重改变没，加载vgg权重测试用
    from keras.utils import plot_model
    plot_model(m, show_shapes=True, to_file='model_segnet.png')
    print(len(m.layers))
    m.summary()
