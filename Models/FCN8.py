from keras.applications import vgg16
from keras.models import Model,Sequential
from keras.layers import Conv2D, Conv2DTranspose, Input,Cropping2D,add


def FCN8_helper(nClasses, image_size):

    assert image_size % 32 == 0

    img_input = Input(shape=( image_size, image_size,3))

    model = vgg16.VGG16(
        include_top=False,
        weights='imagenet',input_tensor=img_input,
        pooling=None,
        classes=1000)
    assert isinstance(model,Model)

    o=Conv2D(filters=4096,kernel_size=(7,7),padding="same",activation="relu",name="fc6")(model.output)
    o = Conv2D(filters=4096, kernel_size=(1, 1), padding="same", activation="relu", name="fc7")(o)
    o = Conv2D(filters=nClasses, kernel_size=(1,1), padding="same", activation="relu",
               name="score_fr")(o) #16

    o=Conv2DTranspose(filters=nClasses,kernel_size=(4,4,),strides=(2,2),padding="valid",activation=None,
                      name="score2")(o) #34

    o=Cropping2D(cropping=((0,2),(0,2)))(o) # 32

    fcn8=Model(inputs=img_input,outputs=o)
    # mymodel.summary()
    return fcn8

def FCN8(nClasses, image_size=512):
    fcn8=FCN8_helper(nClasses, image_size)

    # Conv to be applied on Pool4
    skip_con1 = Conv2D(nClasses, kernel_size=(1, 1), padding="same", activation=None,
                       name="score_pool4")( fcn8.get_layer("block4_pool").output)
    Summed = add(inputs=[skip_con1, fcn8.output])

    x = Conv2DTranspose(nClasses, kernel_size=(4, 4), strides=(2, 2), padding="valid", activation=None,
                        name="score4")(Summed)
    x = Cropping2D(cropping=((0, 2), (0, 2)))(x)

    # Conv to be applied to pool3
    skip_con2 = Conv2D(nClasses, kernel_size=(1, 1), padding="same", activation=None,
                       name="score_pool3")( fcn8.get_layer("block3_pool").output)
    Summed2 = add(inputs=[skip_con2, x])
    Up = Conv2DTranspose(nClasses, kernel_size=(16, 16), strides=(8, 8),
                         padding="valid", activation=None, name="upsample")(Summed2)
    final = Cropping2D(cropping=((0, 8), (0, 8)))(Up)

    mymodel=Model(inputs=fcn8.input,outputs=final)

    return mymodel

if __name__ == '__main__':
    m = FCN8(15,image_size=320)
    from keras.utils import plot_model
    plot_model(m, show_shapes=True, to_file='model.png')
    print(len(m.layers))
