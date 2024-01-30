#!/bin/bash

find ../../../upscaling/KXNet/x2/*.png | xargs -P 1000 -i ./split.py {} --output_folder ./KXNet/x2
find ../../../upscaling/lanczos/x2/*.png | xargs -P 1000 -i ./split.py {} --output_folder ./lanczos/x2
find ../../../upscaling/Real-ESRGAN/x2/*.png | xargs -P 1000 -i ./split.py {} --output_folder ./Real-ESRGAN/x2
find ../../../upscaling/waifu2x/x2/*.png | xargs -P 1000 -i ./split.py {} --output_folder ./waifu2x/x2
find ../../../upscaling/BSRGAN/x2/*.png | xargs -P 1000 -i ./split.py {} --output_folder ./BSRGAN/x2

find ../../../upscaling/KXNet/x4/*.png | xargs -P 1000 -i ./split.py {} --output_folder ./KXNet/x4
find ../../../upscaling/lanczos/x4/*.png | xargs -P 1000 -i ./split.py {} --output_folder ./lanczos/x4
find ../../../upscaling/Real-ESRGAN/x4/*.png | xargs -P 1000 -i ./split.py {} --output_folder ./Real-ESRGAN/x4
find ../../../upscaling/waifu2x/x4/*.png | xargs -P 1000 -i ./split.py {} --output_folder ./waifu2x/x4
find ../../../upscaling/BSRGAN/x4/*.png | xargs -P 1000 -i ./split.py {} --output_folder ./BSRGAN/x4

find ../../../upscaling/src_images_1080/x1/*.png | xargs -P 1000 -i ./split.py {} --output_folder ./src_images_1080/x1
