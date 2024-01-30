#!/bin/bash
base_url="https://avtshare01.rz.tu-ilmenau.de/ai_upscaling/"

wget -c "$base_url/upscaling.tar.xz"
tar -xvf upscaling.tar.xz
wget -c "$base_url/upscaling_detection_models.tar.xz"
tar -xvf upscaling_detection_models.tar.xz
