#!/usr/bin/env python3
import argparse
import sys
import os
import multiprocessing

import skimage.io

def img_show(x):
    skimage.io.imshow(x)
    skimage.io.show()


def img_split(img_filename, folder):
    os.makedirs(folder, exist_ok=True)
    img = skimage.io.imread(img_filename)

    h, w = img.shape[0:2]
    top_left = img[0:h // 2, 0:w // 2, ]

    bn_img = os.path.join(
        folder,
        os.path.splitext(os.path.basename(img_filename))[0]
    )
    i = 0
    for x in range(0, h, 224):
        for y in range(0, w, 224):
            patch = img[x: x + 224, y: y + 224, ]
            if patch.shape[0:2] != (224, 224):
                # skip incomplete borders
                continue
            skimage.io.imsave(bn_img + f"_{i}.png", patch, check_contrast=False)
            i += 1

    print(f"{img_filename} done.")


def main(_):
    # argument parsing
    parser = argparse.ArgumentParser(description='split image in 4 patches',
                                     epilog="stg7 2020",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("image", type=str, nargs="+", help="image to be splitted")
    parser.add_argument("--output_folder", type=str, default="splits", help="folder to store all splitted images")
    parser.add_argument('--cpu_count', type=int, default=multiprocessing.cpu_count() // 2, help='thread/cpu count')

    a = vars(parser.parse_args())

    pool = multiprocessing.Pool(processes=a["cpu_count"])

    params = [(image, a["output_folder"]) for image in a["image"]]
    pool.starmap(img_split, params)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


