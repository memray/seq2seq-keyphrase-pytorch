# seq2seq-keyphrase-pytorch

## **Note: this repository is basically deprecated. Please move to our latest code/data/model release for keyphrase generation at [https://github.com/memray/OpenNMT-kpg-release](https://github.com/memray/OpenNMT-kpg-release).**




Current code is developed on PyTorch 0.4, not sure if it works on other versions.

A subset of data (20k docs) is provided [here](https://drive.google.com/open?id=1Jh8Suuk6sTKuK-mbpvU5KfiQKi9zAGar) for you to test the code. Unzip and place it to data/.
  
If you need to train on the whole kp20k dataset, download the [json data](https://drive.google.com/file/d/1ZTQEGZSq06kzlPlOv4yGjbUpoDrNxebR/view) and run `preprocess.py` first. No trained model will be released in the near future.

**Update**
I will not be updating this repo for a while. But please see the information below to help you run the code. Some
Some test datasets in JSON format: [download](https://drive.google.com/open?id=1jiPSgTO6ofF9QSYjBplCVHMf12_p7QXr)
 - **preprocess.py**: entry for preprocessing datasets in JSON format.
 - **train.py**: entry for training models.
 - **predict.py**: entry for generating phrases with well-trained models (checkpoints).

You can refer to these [scripts](https://github.com/memray/seq2seq-keyphrase-pytorch/tree/master/script) as examples.

Note that duplicate papers that appear in popular test datasets (e.g. Inspec, SemEval) are also included in the KP20k training dataset. Please be sure to remove them before training.
