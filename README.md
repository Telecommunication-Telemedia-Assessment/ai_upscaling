# ai_upscaling
Repository for AI-based up-scaling evaluation.

We evaluated the following AI-based up-scaling methods:

* BSRGAN: https://github.com/cszn/BSRGAN
* KXNet: https://github.com/jiahong-fu/KXNet
* Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
* waifu2x: https://github.com/nihui/waifu2x-ncnn-vulkan


## Download 
To download the images and pre-trained models, you need to run `./download.sh` (check this script for the baseurl of the hosted data).
This requires `xz-utils` to be installed under Ubuntu.



## Structure

* `upscaling`: images used for the subjective test (including the 1080p reference images and up-scaled variants), the test was conducted using [AVrateVoyager](https://github.com/Telecommunication-Telemedia-Assessment/AVrateVoyager).
* `evaluation`: evaluation scripts (jupyter notebook required) and subjective annotations (`/subjective/*` or `mos.csv`)
* `upscaling_features` calculated signal and other features



## Note
The DNN experiments are modified variants of the following two repositories:

* [sophoappeal_appeal_prediction_dnn_transferlearning](https://github.com/Telecommunication-Telemedia-Assessment/sophoappeal_appeal_prediction_dnn_transferlearning)
* [sophoappeal_rule_prediction_extension](https://github.com/Telecommunication-Telemedia-Assessment/sophoappeal_rule_prediction_extension)

additional code for prediction (and not only training and evaluation) can be found in the corresponding repositories and must be adjusted.

Each of the DNN experiments needs specifically prepared data, where we provide scripts for the creation in the corresponding data folder.

## Requirements

The provided software is tested under Ubuntu 22.04 and 23.10.

* python3, python3-pip, jupyter notebook/lab




## Acknowledgments

If you use this software or data in your research, please include a link to the repository and reference the following paper.

```bibtex
@article{goering2024aiupscaling,
  title={Appeal prediction for AI up-scaled Images},
  author={Steve G\"oring and Rasmus Merten and Alexander Raake},
  note={to appear},
}

```

## License
GNU General Public License v3. See [LICENSE.md](./LICENSE.md) file in this repository.