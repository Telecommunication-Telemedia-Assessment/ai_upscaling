#!/usr/bin/env python3
import os

import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras


def get_df(basedir):
    data_dir = "data_cc_small"
    df = pd.read_csv(os.path.join(basedir, f"{data_dir}/mos.csv")) #.head(1000)

    df["image"] = df["stimuli_file"].apply(
        lambda x: os.path.join(basedir, f"{data_dir}/" + x.replace("./stimuli/", ""))
    )
    df["rating"] = (df["mos"] - 1) / 4
    print(df["rating"].min(), df["rating"].max())
    print(df.head())
    return df


models = {
    "Xception": {
        "__init__": keras.applications.Xception,
        "preprocess": tf.keras.applications.xception.preprocess_input,
    },
    "VGG19": {
        "__init__": keras.applications.VGG19,
        "preprocess": tf.keras.applications.vgg19.preprocess_input,
    },
    "VGG16": {
        "__init__": keras.applications.VGG16,
        "preprocess": tf.keras.applications.vgg16.preprocess_input,
    },
    "MobileNetV2": {
        "__init__": keras.applications.MobileNetV2,
        "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input,
    },
    "MobileNet": {
        "__init__": keras.applications.MobileNet,
        "preprocess": tf.keras.applications.mobilenet.preprocess_input,
    },
    "ResNet50": {
        "__init__": keras.applications.ResNet50,
        "preprocess": tf.keras.applications.resnet.preprocess_input,
    },
    "ResNet101": {
        "__init__": keras.applications.ResNet101,
        "preprocess": tf.keras.applications.resnet.preprocess_input,
    },
    "ResNet152": {
        "__init__": keras.applications.ResNet152,
        "preprocess": tf.keras.applications.resnet.preprocess_input,
    },
    "ResNet50V2": {
        "__init__": keras.applications.ResNet50V2,
        "preprocess": tf.keras.applications.resnet_v2.preprocess_input,
    },
    "ResNet101V2": {
        "__init__": keras.applications.ResNet101V2,
        "preprocess": tf.keras.applications.resnet_v2.preprocess_input,
    },
    "ResNet152V2": {
        "__init__": keras.applications.ResNet152V2,
        "preprocess": tf.keras.applications.resnet_v2.preprocess_input,
    },
    "InceptionV3": {
        "__init__": keras.applications.InceptionV3,
        "preprocess": tf.keras.applications.inception_v3.preprocess_input,
    },
    "InceptionResNetV2": {
        "__init__": keras.applications.InceptionResNetV2,
        "preprocess": tf.keras.applications.inception_resnet_v2.preprocess_input,
    },
    "DenseNet121": {
        "__init__": keras.applications.DenseNet121,
        "preprocess": tf.keras.applications.densenet.preprocess_input,
    },
    "DenseNet169": {
        "__init__": keras.applications.DenseNet169,
        "preprocess": tf.keras.applications.densenet.preprocess_input,
    },
    "DenseNet201": {
        "__init__": keras.applications.DenseNet201,
        "preprocess": tf.keras.applications.densenet.preprocess_input,
    },
    "NASNetMobile": {
        "__init__": keras.applications.NASNetMobile,
        "preprocess": tf.keras.applications.nasnet.preprocess_input,
    },

    # NASNetLarge excluded because it requires a fixed input
    #"NASNetLarge": {
    #    "__init__": keras.applications.NASNetLarge,
    #    "preprocess": tf.keras.applications.nasnet.preprocess_input,
    #},
}