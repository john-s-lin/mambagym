#!/bin/bash

conda activate mambagym

python train.py --full_dose_path "Preprocessed_256x256/256/Full Dose" --quarter_dose_path "Preprocessed_256x256/256/Quarter Dose" --path_to_save "output/"