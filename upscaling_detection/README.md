# detection of which upscaling method has been used

## approach
based on multi classification using transfer learned DNNs, similar to photo rule prediction approach, see https://github.com/Telecommunication-Telemedia-Assessment/sophoappeal_rule_prediction_extension

## requirements
use conda to install the exported envirnment 

## experiment 1
using the images as they are --> failed, due to the fact that all images are rescaled to 224x244 for the DNNs

## experiment 2
using 224x224 patches of images

preparation: `data/upscalers_splits/split_all.sh` must be executed to create all patches

then run `./evaluate_models_multiclass.py --data data/upscalers_splits --models_folder models/upscalers_splits --results_folder results/upscalers_splits
` for all included models, may take some time and required GPUs


