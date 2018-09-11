import logging
import numpy as np
import torch
from gensen.gensen import GenSenSingle


class SentenceEncoder(object):
    """Gets sentence representations from a pretrained sentence encoder"""

    def __init__(self, gensen_data_path, vocab_expansion=None, cuda=False):
        self.gensen_data_path = gensen_data_path
        self.gensen = GenSenSingle(model_folder=self.gensen_data_path + 'models',
                                   filename_prefix='nli_large_bothskip_parse',
                                   pretrained_emb=self.gensen_data_path + 'embedding/glove.840B.300d.h5',
                                   cuda=cuda)
        if vocab_expansion is not None:
            self.gensen.vocab_expansion(vocab_expansion)
        logging.info("Finished loading pretrained sentence encoder...")

    def get_representations(self, list_of_sentences, return_numpy=False):
        # Sentences need to be lowercased.
        list_of_sentences = [" ".join(sent) for sent in list_of_sentences]
        list_of_sentences = [sent.lower() for sent in list_of_sentences]
        reps_h, reps_h_t = self.gensen.get_representation(list_of_sentences, pool='last', return_numpy=return_numpy, add_start_end=False)
        # reps_h: batch x sent len x 2048
        # reps_h_t: batch x 2048
        return reps_h, reps_h_t

    def get_prefix_represenatatons(self, list_of_sentences, return_numpy=False):
        # "<pad>" is the padding token in gensen
        maxlen = max(map(len, list_of_sentences))
        for i in range(len(list_of_sentences)):
            list_of_sentences[i] = list_of_sentences[i] + ["<pad>"] * (maxlen - len(list_of_sentences[i]))
        res = []
        for i in range(maxlen):
            tmp_inp = [sent[: i + 1] for sent in list_of_sentences]
            res.append(self.get_representations(tmp_inp, return_numpy=return_numpy)[1])
        if return_numpy:
            return np.stack(res, 1)
        else:
            return torch.stack(res, 1)


if __name__ == '__main__':
    gensen = SentenceEncoder("./data/", vocab_expansion=None)
    # Sentences need to be lowercased.
    sentences = [
        'hello world .',
        'the quick brown fox jumped over the lazy dog .',
        'this is a sentence .',
        'the white cat is sleeping on the ground .'
    ]

    _, reps_h_t = gensen.get_representations(sentences, return_numpy=True)

    from scipy.spatial import distance

    for i in range(3):
        for j in range(i + 1, 4):
            c = 1.0 - distance.cosine(reps_h_t[i], reps_h_t[j])
            print(i, j, c)

