#!/usr/bin/env python3
import os
import sys
import argparse
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# model specific imports
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.densenet import preprocess_input

from utils import get_df
from utils import models

parser = argparse.ArgumentParser(
    description='retrain script',
    epilog="stg7 2022",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--model",
    type=str,
    default="DenseNet121",
    choices=set(models.keys()),
    help="model to use"
)
parser.add_argument("--no_gpu", action="store_true", help="do not use gpu")
parser.add_argument("--debug", action="store_true", help="debug run")
parser.add_argument("--basedir", type=str, default=".", help="basedir of training data")


a = vars(parser.parse_args())
modelname = a["model"]

print(f"use model {modelname}")
# this must be configured here, otherwise some errors occur
GPU = not a["no_gpu"]
DEBUG = a["debug"]
BASEDIR = a["basedir"]

if GPU:
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)


strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))



def make_generator(df, image_size, batch_size):
    _igen = ImageDataGenerator(
            rescale=1./255,
            vertical_flip=True,
            brightness_range=(0.2, 0.8),
            zoom_range=0.1,
    )

    image_gen = _igen.flow_from_dataframe(
        dataframe=df,
        directory=None,
        x_col="image",
        y_col="rating",
        weight_col=None,
        target_size=(image_size, image_size),
        color_mode="rgb",
        classes=None,
        class_mode="raw",
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        subset=None,
        interpolation="nearest"
    )
    return image_gen


def get_train_val_generators(image_size, batch_size, basedir):

    df = get_df(basedir)

    dtrain, dval = train_test_split(df, test_size=0.25, random_state=42)

    print(dtrain.head())
    print(dval.head())

    print(len(dtrain), len(dval))

    image_gen = make_generator(dtrain, image_size, batch_size)
    image_gen_validation = make_generator(dval, image_size, batch_size)

    return image_gen, image_gen_validation


def create_model(modelname):

    # create the base pre-trained model
    base_model = models[modelname]["__init__"](weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output

    # TODO: CHECK THE NEXT TWO STEPS
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 1 class
    predictions = Dense(1, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    model.summary()

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.MeanAbsoluteError()
        ]
    )

    return model


def main(args):
    batch_size = 10

    if GPU:
        batch_size = 256 # +64, #128, #32,

    image_size = 224
    num_epochs = 50

    image_gen, image_gen_validation = get_train_val_generators(image_size, batch_size, BASEDIR)


    with strategy.scope():
        model = create_model(modelname)

    checkpoint_filepath = "./tmp_retrain/checkpoint/" + modelname

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_mse",
        save_best_only=True,
        save_weights_only=True
    )


    steps_per_epoch = len(image_gen)
    validation_steps = len(image_gen_validation)
    if DEBUG:
        num_epochs = 2
        steps_per_epoch = 2
        validation_steps = 2

    history = model.fit(
        x=image_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        validation_data=image_gen_validation,
        validation_steps=validation_steps,
        callbacks=[checkpoint_callback],
        verbose=1
    )

    print("fitting done, store results")

    # store history
    dh = pd.DataFrame(history.history)
    ax = dh.plot(kind="line")
    ax.get_figure().savefig(modelname + "_hist_retrain.pdf")
    dh.to_csv(modelname + "_hist_retrain.csv", index=False)

    # load stored model and predict values for validation set
    model.load_weights(checkpoint_filepath)

    predictions = []
    truth = []
    i = 0
    for x, y in image_gen_validation:
        if i >= len(image_gen_validation):
            break
        i += 1
        print(f"predict batch {i}/{len(image_gen_validation)}")
        pred = model.predict(x)
        predictions.extend(pred.flatten())
        truth.extend(y.flatten())
        if DEBUG:
            break

    dr = pd.DataFrame({
        "truth": truth,
        "prediction": predictions
    })
    print(dr.corr())
    os.makedirs("results", exist_ok=True)

    dr.to_csv("results/" + modelname + "_predictions.csv", index=False)
    ax = dr.plot(x="truth", y="prediction", kind="scatter")
    ax.get_figure().savefig("results/" + modelname + "_scatter_retrain.pdf")

    print(max(predictions), min(predictions))
    print("Model:{}, P:{}, K:{}, S:{}".format(
        modelname,
        dr.corr(method="pearson"),
        dr.corr(method="kendall"),
        dr.corr(method="spearman"),
        )
    )

    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
