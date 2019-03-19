#!/usr/bin/env bash
python -m pykp.data.remove_duplicates_multiprocess -train_file source_data/mag/mag_training_small.json -datatype mag -n_jobs 60


nohup python -m pykp.data.remove_duplicates_multiprocess -train_file source_data/mag/mag_training_small.json -datatype mag -n_jobs 60 > source_data/nohup_mag.txt &

nohup python -m pykp.data.remove_duplicates_multiprocess -train_file source_data/mag/mag_training.json -datatype mag -n_jobs 60 > source_data/nohup_mag.txt &