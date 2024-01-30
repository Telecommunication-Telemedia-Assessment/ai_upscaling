#!/usr/bin/tcsh

if ("$1" == "") then
  echo "script requires a parameter"
  exit 0
endif

echo "start training"
date
conda activate ups
# ./evaluate_models_multiclass.py --model_idx_start $1 --model_idx_end $1  # rule prediction
./evaluate_models_multiclass.py --model_idx_start $1 --model_idx_end $1 --data data/upscalers_splits --models_folder models/upscalers_splits --results_folder results/upscalers_splits
echo "training done"
date
