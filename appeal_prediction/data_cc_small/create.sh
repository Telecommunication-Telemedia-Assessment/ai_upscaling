#!/bin/bash 

cp ../../evaluation/mos.csv .
cp -r ../../upscaling/* .

find */ -name "*.png" | xargs -i ./center_crop.sh {}