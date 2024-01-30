#!/bin/bash
mogrify -gravity center -crop 224x224+0+0 +repage "$1"