#!/usr/bin/env python3
"""
the following code is partially based on https://keras.io/guides/transfer_learning/

code style: black -l 120 <filename>
"""
import os
import argparse
import json
import sys
#os.environ["CUDA_VISIBLE_DEVICES"]="3"

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from tqdm import tqdm


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


def build_transfer_learning_model(modelname, num_of_classes):

    assert modelname in models
    # get model
    model = models[modelname]

    input_shape = (224, 224, 3)

    base_model = model["__init__"](
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=input_shape,
        include_top=False,
    )  # Do not include the ImageNet classifier at the top.

    # Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = keras.Input(shape=input_shape)
    # we need to use the DNN specific preprocessing of the image
    x = model["preprocess"](inputs)
    x = base_model(x, training=False)
    #x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(num_of_classes, activation = "softmax")(x)
    # based on https://www.dlology.com/blog/how-to-choose-last-layer-activation-and-loss-function/
    model = keras.Model(inputs, outputs, name="cust_" + modelname)
    return model


def check_model_creation():
    # check model access
    print(build_transfer_learning_model("DenseNet201").summary())


def read_images(directory, subset, input_shape, batch_size=32, split=0.1):
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=input_shape[0:2],
        shuffle=True,
        seed=42,
        validation_split=split,  # 10% validation
        subset=subset,
        interpolation="bilinear",
    )


def train_and_evaluate_model(modelname, input_shape, training, validation, results_folder, models_folder, num_of_classes, epochs=100):
    model = build_transfer_learning_model(modelname, num_of_classes)
    model.compile(
        optimizer='adam',
        loss=tf.losses.CategoricalCrossentropy(), # from_logits=True),
        metrics=['accuracy']
    )

    checkpoint = ModelCheckpoint(
        os.path.join(models_folder, modelname + "_best_model.hdf5"),
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        mode="auto",
        save_freq="epoch",
    )

    res = model.fit(
        training,
        validation_data=validation,
        epochs=epochs,
        callbacks=[checkpoint]
    )

    print("run prediction")

    def get_label(x):
        for class_name in training.class_names:
            if class_name in x:
                return class_name

    y_pred = []
    y_truth = []
    validation_files = []
    for image_filename in sorted(validation.file_paths):
        image = img_to_array(load_img(
            image_filename,
            color_mode='rgb',
            target_size=input_shape,
            interpolation='bilinear'
        ))
        y_truth.append(get_label(image_filename))
        y_pred.append(res.model.predict(np.array([image])))
        validation_files.append(image_filename)

    y_truth = [training.class_names.index(x) for x in np.array(y_truth).flatten()]
    y_pred = [float(x) for x in np.array(y_pred).flatten()]
    print("save results")
    result = res.history
    result["model"] = modelname
    result["classnames"] = training.class_names
    result["pred"] = y_pred
    result["truth"] = y_truth
    result["validation_files"] = validation_files

    with open(os.path.join(results_folder, modelname + ".json"), "w") as xfp:
        json.dump(result, xfp, sort_keys=True, indent=4)


def check_images(path, input_shape):
    print("check images")
    problematic_images = []
    image_read_errors = []
    imgs = read_images(path, None, input_shape, batch_size=128, split=0)
    for image_filename in tqdm(sorted(imgs.file_paths)):
        try:
            image = img_to_array(load_img(
                image_filename,
                color_mode='rgb',
                target_size=input_shape,
                interpolation='bilinear'
            ))
        except Exception as e:
            print(e)
            print(image_filename)
            problematic_images.append(image_filename)
            image_read_errors.append(str(e))
    if len(image_read_errors) > 0:
        print("\n".join(problematic_images))
        print("\n".join(image_read_errors))
        sys.exit()
    print("images checked")
    return


def main(_):
    # argument parsing
    parser = argparse.ArgumentParser(
        description="dnn evaluation for multiple image aesthetic rules",
        epilog="stg7, rasmus 2022",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data", default="data/upscalers", type=str, help="data to be used")
    parser.add_argument("--models_folder", default="models/upscalers", type=str, help="folder to store best models")
    parser.add_argument("--results_folder", default="results/upscalers", type=str, help="folder to store results of best models")
    parser.add_argument("--epochs", default=150 , type=int, help="number of epochs per model")
    parser.add_argument("--no_gpu", action="store_true", help="do not use gpu")
    parser.add_argument("--batch_size", default=512, type=int, help="size of batches")
    parser.add_argument("--check_images", action="store_true", help="check input images first")
    parser.add_argument("--model_idx_start", default=0, choices=list(range(len(models))), type=int, help="start index of the model which should be used")
    parser.add_argument("--model_idx_end", default=len(models) - 1, choices=list(range(len(models))), type=int, help="end index of the model which should be used")


    a = vars(parser.parse_args())

    # this must be configured here, otherwise some errors ocurr
    print("detected GPUS:")
    print("\n".join(map(str, tf.config.list_physical_devices("GPU"))))
    gpu_devices = tf.config.list_physical_devices("GPU")
    GPU = not a["no_gpu"] and len(gpu_devices) > 0

    if GPU:
        physical_devices = tf.config.list_physical_devices('GPU')
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)

    batch_size = a["batch_size"]

    input_shape = (224, 224, 3)

    if a["check_images"]:
        check_images(a["data"], input_shape)

    training = read_images(a["data"], "training", input_shape, batch_size)
    validation = read_images(a["data"], "validation", input_shape, batch_size)

    found_classes = len(training.class_names)

    print(f"training and validation batches: {len(training)},{len(validation)}")
    print(f"batch size: {batch_size}")
    print(f"input_shape: {input_shape}")

    os.makedirs(a["results_folder"], exist_ok=True)
    os.makedirs(a["models_folder"], exist_ok=True)

    for modelname in sorted(models)[a["model_idx_start"]:a["model_idx_end"] + 1]:
        print(modelname)
        train_and_evaluate_model(
            modelname,
            input_shape,
            training,
            validation,
            a["results_folder"],
            a["models_folder"],
            found_classes,
            epochs=a["epochs"]
        )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
