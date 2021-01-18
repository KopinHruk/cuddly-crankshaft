# Module with functions to work with model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau


def create_model(image_size=64, weights_path='weights/bengalimodal.h5', pretrained=True):
    """
    Description
    -----------
    Creates model from https://www.kaggle.com/amanmishra4yearbtech/bengali-classification-quick-implementation


    Parameters
    ----------
    image_size: int
        Image size to use model with.
    weights_path: str
        Path to model's weights. Required if pretrained=True.
    pretrained: bool:
        Use pretrained weights or no.


    Returns
    -------
    model: tensorflow.keras.models.Model
        CNN model
    """

    inputs = Input(shape=(image_size, image_size, 1))

    model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu',
                   input_shape=(image_size, image_size, 1))(inputs)
    model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
    model = Dropout(rate=0.25)(model)

    model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = Dropout(rate=0.25)(model)

    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = Dropout(rate=0.2)(model)

    model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=256, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.20)(model)
    model = Dropout(rate=0.25)(model)

    model = Flatten()(model)
    model = Dense(512, activation="relu", name='dense_')(model)
    model = Dropout(rate=0.25)(model)
    dense = Dense(256, activation="relu", name='dense_1')(model)

    head_root = Dense(168, activation='softmax', name='dense_2')(dense)
    head_vowel = Dense(11, activation='softmax', name='dense_3')(dense)
    head_consonant = Dense(7, activation='softmax', name='dense_4')(dense)

    model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])  # 3 outputs one for each
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    if pretrained:
        model.load_weights(weights_path)

    return model


# Model's callbacks
lr_reduction_root = ReduceLROnPlateau(monitor='dense_2_accuracy',
                                      patience=3,
                                      verbose=1,
                                      factor=0.5,
                                      min_lr=0.00001)

lr_reduction_vowel = ReduceLROnPlateau(monitor='dense_3_accuracy',
                                       patience=3,
                                       verbose=1,
                                       factor=0.5,
                                       min_lr=0.00001)

lr_reduction_consonant = ReduceLROnPlateau(monitor='dense_4_accuracy',
                                           patience=3,
                                           verbose=1,
                                           factor=0.5,
                                           min_lr=0.00001)
