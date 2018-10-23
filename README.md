# seq2seq-keyphrase-pytorch

Current code is developed on PyTorch 0.4, not sure if it works on other versions.

A subset of data (20k docs) is provided [here](https://drive.google.com/open?id=1Jh8Suuk6sTKuK-mbpvU5KfiQKi9zAGar) for you to test the code. Unzip and place it to data/.
  
If you need to train on the whole kp20k dataset, download the [json data](https://drive.google.com/file/d/1ZTQEGZSq06kzlPlOv4yGjbUpoDrNxebR/view) and run `preprocess.py` first. No trained model will be released in the near future.

Note that duplicate papers that appear in popular test datasets (e.g. Inspec, SemEval) are also included in the release. Please be sure to remove them before training.