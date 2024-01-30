#!/bin/bash
convert "$1" -crop 224x224 "$(basename $1 .png)slice-%02d.png"