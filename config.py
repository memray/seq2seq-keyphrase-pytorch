import logging

import sys

import time


def init_logging(logfile):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S'   )

    fh = logging.FileHandler(logfile)
    # ch = logging.StreamHandler()
    ch = logging.StreamHandler(sys.stdout)

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # fh.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    logging.getLogger().addHandler(ch)
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """
    # Embedding Options
    parser.add_argument('-word_vec_size', type=int, default=300,
                        help='Word embedding for both.')

    parser.add_argument('-position_encoding', action='store_true',
                        help='Use a sin to mark relative words positions.')
    parser.add_argument('-share_decoder_embeddings', action='store_true',
                        help='Share the word and out embeddings for decoder.')
    parser.add_argument('-share_embeddings', action='store_true',
                        help="""Share the word embeddings between encoder
                         and decoder.""")

    # RNN Options
    parser.add_argument('-encoder_type', type=str, default='rnn',
                        choices=['rnn', 'brnn', 'mean', 'transformer', 'cnn'],
                        help="""Type of encoder layer to use.""")
    parser.add_argument('-decoder_type', type=str, default='rnn',
                        choices=['rnn', 'transformer', 'cnn'],
                        help='Type of decoder layer to use.')

    parser.add_argument('-enc_layers', type=int, default=1,
                        help='Number of layers in the encoder')
    parser.add_argument('-dec_layers', type=int, default=1,
                        help='Number of layers in the decoder')

    parser.add_argument('-rnn_size', type=int, default=512,
                        help='Size of LSTM hidden states')
    parser.add_argument('-input_feed', type=int, default=1,
                        help="""Feed the context vector at each time step as
                        additional input (via concatenation with the word
                        embeddings) to the decoder.""")

    parser.add_argument('-rnn_type', type=str, default='LSTM',
                        choices=['LSTM', 'GRU'],
                        help="""The gate type to use in the RNNs""")
    # parser.add_argument('-residual',   action="store_true",
    #                     help="Add residual connections between RNN layers.")

    parser.add_argument('-bidirectional',
                        action = "store_true",
                        help="whether the encoder is bidirectional")

    parser.add_argument('-brnn_merge', default='concat',
                        choices=['concat', 'sum'],
                        help="Merge action for the bidir hidden states")

    parser.add_argument('-context_gate', type=str, default=None,
                        choices=['source', 'target', 'both'],
                        help="""Type of context gate to use.
                        Do not select for no context gate.""")

    # Attention options
    parser.add_argument('-global_attention', type=str, default='general',
                        choices=['dot', 'general', 'mlp'],
                        help="""The attention type to use:
                        dotprot or general (Luong) or MLP (Bahdanau)""")

    # Genenerator and loss options.
    parser.add_argument('-copy_attn', action="store_true",
                        help='Train copy attention layer.')
    parser.add_argument('-copy_attn_force', action="store_true",
                        help='When available, train to copy.')
    parser.add_argument('-coverage_attn', action="store_true",
                        help='Train a coverage attention layer.')
    parser.add_argument('-lambda_coverage', type=float, default=1,
                        help='Lambda value for coverage.')


def preprocess_opts(parser):
    # Dictionary Options
    parser.add_argument('-vocab_size', type=int, default=50000,
                        help="Size of the source vocabulary")

    parser.add_argument('-words_min_frequency', type=int, default=0)

    # Length filter options
    parser.add_argument('-max_src_seq_length', type=int, default=300,
                        help="Maximum source sequence length")
    parser.add_argument('-min_src_seq_length', type=int, default=20,
                        help="Minimum source sequence length")
    parser.add_argument('-max_trg_seq_length', type=int, default=8,
                        help="Maximum target sequence length to keep.")
    parser.add_argument('-min_trg_seq_length', type=int, default=None,
                        help="Minimun target sequence length to keep.")

    # Truncation options
    parser.add_argument('-src_seq_length_trunc', type=int, default=None,
                        help="Truncate source sequence length.")
    parser.add_argument('-trg_seq_length_trunc', type=int, default=None,
                        help="Truncate target sequence length.")

    # Data processing options
    parser.add_argument('-shuffle', type=int, default=1,
                        help="Shuffle data")
    parser.add_argument('-lower', default=True,
                        action = 'store_true', help='lowercase data')

    # Options most relevant to summarization
    parser.add_argument('-dynamic_dict', default=True,
                        action='store_true', help="Create dynamic dictionaries (for copy)")

def train_opts(parser):
    # Model loading/saving options
    parser.add_argument('-data', required=True,
                        help="""Path prefix to the ".train.pt" and
                        ".valid.pt" file path from preprocess.py""")
    parser.add_argument('-vocab', required=True,
                        help="""Path prefix to the ".vocab.pt"
                        file path from preprocess.py""")

    parser.add_argument('-save_model', default='model',
                        help="""Model filename (the model will be saved as
                        <save_model>_epochN_PPL.pt where PPL is the
                        validation perplexity""")
    parser.add_argument('-train_from', default='', type=str,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model's state_dict.""")
    # GPU
    parser.add_argument('-gpuid', default=0, nargs='+', type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-seed', type=int, default=9527,
                        help="""Random seed used for the experiments
                        reproducibility.""")

    # Init options
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')
    parser.add_argument('-param_init', type=float, default=0.1,
                        help="""Parameters are initialized over uniform distribution
                        with support (-param_init, param_init).
                        Use 0 to not use initialization""")

    # Pretrained word vectors
    parser.add_argument('-pre_word_vecs_enc',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the encoder side.
                        See README for specific formatting instructions.""")
    parser.add_argument('-pre_word_vecs_dec',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the decoder side.
                        See README for specific formatting instructions.""")
    # Fixed word vectors
    parser.add_argument('-fix_word_vecs_enc',
                        action='store_true',
                        help="Fix word embeddings on the encoder side.")
    parser.add_argument('-fix_word_vecs_dec',
                        action='store_true',
                        help="Fix word embeddings on the encoder side.")

    # Optimization options
    parser.add_argument('-batch_size', type=int, default=256,
                        help='Maximum batch size')
    parser.add_argument('-batch_workers', type=int, default=1,
                        help='Number of workers for generating batches')
    parser.add_argument('-epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('-optim', default='adam',
                        choices=['sgd', 'adagrad', 'adadelta', 'adam'],
                        help="""Optimization method.""")
    parser.add_argument('-max_grad_norm', type=float, default=5,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to
                        max_grad_norm""")
    parser.add_argument('-dropout', type=float, default=0.5,
                        help="Dropout probability; applied in LSTM stacks.")
    parser.add_argument('-truncated_decoder', type=int, default=0,
                        help="""Truncated bptt.""")
    # learning rate
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Starting learning rate.
                        Recommended settings: sgd = 1, adagrad = 0.1,
                        adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past
                        start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=8,
                        help="""Start decaying every epoch after and including this
                        epoch""")
    parser.add_argument('-start_checkpoint_at', type=int, default=0,
                        help="""Start checkpointing every epoch after and including
                        this epoch""")
    parser.add_argument('-decay_method', type=str, default="",
                        choices=['noam'], help="Use a custom decay rate.")
    parser.add_argument('-warmup_steps', type=int, default=4000,
                        help="""Number of warmup steps for custom decay.""")

    parser.add_argument('-run_valid_every', type=int, default=1000,
                        help="Run validation test at this interval (every run_valid_every epochs)")
    parser.add_argument('-early_stop_tolerance', type=int, default=10,
                        help="Stop training if it doesn't improve any more for serveral epochs")

    timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))

    parser.add_argument('-report_every', type=int, default=50,
                        help="Print stats at this interval.")

    parser.add_argument('-exp_path', type=str, default="exp/stackexchange.%s" % timemark,
                        help="Path of experiment output/log/checkpoint.")

    parser.add_argument('-exp', type=str, default="stackexchange",
                        help="Name of the experiment for logging.")


def predict_opts(parser):
    parser.add_argument('-test_data',   required=True,
                        help="""Source sequence to decode (one line per
                        sequence)""")
    parser.add_argument('-save_data', required=True,
                        help="Output file for the prepared test data")
    parser.add_argument('-model_path', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-vocab', required=True,
                        help="""Path prefix to the ".vocab.pt"
                        file path from preprocess.py""")
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size',  type=int, default=3,
                        help='Beam size')
    parser.add_argument('-max_sent_length', type=int, default=4,
                        help='Maximum sentence length.')
    parser.add_argument('-replace_unk', action="store_true",
                        help="""Replace the generated UNK tokens with the
                        source token that had highest attention weight. If
                        phrase_table is provided, it will lookup the
                        identified source token and give the corresponding
                        target token. If it is not provided(or the identified
                        source token does not exist in the table) then it
                        will copy the source token""")
    parser.add_argument('-verbose', action="store_true",
                        help='Print scores and predictions for each sentence')
    parser.add_argument('-attn_debug', action="store_true",
                        help='Print best attn for each word')
    parser.add_argument('-dump_beam', type=str, default="",
                        help='File to dump beam information to.')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    # GPU
    parser.add_argument('-gpuid', default=0, nargs='+', type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-seed', type=int, default=9527,
                        help="""Random seed used for the experiments
                        reproducibility.""")

    # batch setting
    parser.add_argument('-batch_size', type=int, default=2,
                        help='Maximum batch size')
    parser.add_argument('-batch_workers', type=int, default=1,
                        help='Number of workers for generating batches')
