import pykp.IO
import argparse
import codecs
import json
import re
import torch
import torchtext
import config

parser = argparse.ArgumentParser( description='preprocess.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# **Preprocess Options**
parser.add_argument('-config', help="Read options from this file")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

config.preprocess_opts(parser)

opt = parser.parse_args()
# torch.manual_seed(opt.seed)

def main():

    print("Loading train/valid to disk: %s" % (opt.save_data + '.train_valid.pt'))
    data_dict = torch.load( open(opt.save_data + '.train_valid.pt', 'rb'))
    # print(data_dict.keys())
    train = data_dict['train']
    print(train)




    # model = torch.load(open(opt.save_data+""))
    ## question - to phrase
    ## how many questions
    # how many phrase unique



if __name__ == "__main__":
    main()