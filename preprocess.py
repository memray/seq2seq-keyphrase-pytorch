import yaml
import torch
import pykp.io


with open("config.yaml") as reader:
    config = yaml.safe_load(reader)

config['train_path'] = 'data/%s/%s_training.json' % (config['general']['dataset'], config['general']['dataset'])
config['valid_path'] = 'data/%s/%s_validation.json' % (config['general']['dataset'], config['general']['dataset'])
config['test_path'] = 'data/%s/%s_testing.json' % (config['general']['dataset'], config['general']['dataset'])
config['save_data']  = 'data/%s/%s' % (config['general']['dataset'], config['general']['dataset'])

def main():
    '''
    Load and process training data
    '''
    # load keyphrase data from file, each data example is a pair of (src_str, [kp_1, kp_2 ... kp_m])

    if config['general']['dataset'] == 'kp20k':
        src_fields = ['title', 'abstract']
        trg_fields = ['keyword']
    elif config['general']['dataset'] == 'stackexchange':
        src_fields = ['title', 'question']
        trg_fields = ['tags']
    elif config['general']['dataset'] == 'twacg':
        src_fields = ['title', 'observation']
        trg_fields = ['admissible_commands']
    else:
        raise Exception('Unsupported dataset name=%s' % config['general']['dataset'])

    print("Loading training data...")
    src_trgs_pairs = pykp.io.load_json_data(config['train_path'], name=config['general']['dataset'], src_fields=src_fields, trg_fields=trg_fields, trg_delimiter=';')

    print("Processing training data...")
    tokenized_train_pairs = pykp.io.tokenize_filter_data(src_trgs_pairs, tokenize=pykp.io.copyseq_tokenize, config=config, valid_check=True)

    print("Building Vocab...")
    word2id, id2word = pykp.io.build_vocab(tokenized_train_pairs)
    print('Vocab size = %d' % len(id2word))

    print("Building training...")
    train_dump = pykp.io.build_dataset(tokenized_train_pairs, word2id, id2word, config)
    print("Dumping train dump to disk: %s" % (config['save_data'] + '.train_dump.pt'))
    torch.save(train_dump, open(config['save_data'] + '.train_dump.pt', 'wb'))
    len_train_dump = len(train_dump)
    train_dump = None

    '''
    Load and process validation data
    '''
    print("Loading validation data...")
    src_trgs_pairs = pykp.io.load_json_data(config['valid_path'], name=config['general']['dataset'], src_fields=src_fields, trg_fields=trg_fields, trg_delimiter=';')

    print("Processing validation data...")
    tokenized_valid_pairs = pykp.io.tokenize_filter_data(src_trgs_pairs, tokenize=pykp.io.copyseq_tokenize, config=config, valid_check=True)
    print("Building validation...")
    valid_dump = pykp.io.build_dataset(tokenized_valid_pairs, word2id, id2word, config, include_original=True)

    '''
    Load and process test data
    '''
    print("Loading test data...")
    src_trgs_pairs = pykp.io.load_json_data(config['test_path'], name=config['general']['dataset'], src_fields=src_fields, trg_fields=trg_fields, trg_delimiter=';')

    print("Processing test data...")
    tokenized_test_pairs = pykp.io.tokenize_filter_data(src_trgs_pairs, tokenize=pykp.io.copyseq_tokenize, config=config, valid_check=True)
    print("Building testing...")
    test_dump = pykp.io.build_dataset(tokenized_test_pairs, word2id, id2word, config, include_original=True)

    print('#pairs of train_dump = %d' % len_train_dump)
    print('#pairs of valid_dump = %d' % len(valid_dump))
    print('#pairs of test_dump  = %d' % len(test_dump))

    print("***************** Source Length Statistics ******************")
    len_counter = {}
    for src_tokens, trgs_tokens in tokenized_train_pairs:
        len_count = len_counter.get(len(src_tokens), 0) + 1
        len_counter[len(src_tokens)] = len_count
    sorted_len = sorted(len_counter.items(), key=lambda x:x[0], reverse=True)

    for len_, count in sorted_len:
        print('%d,%d' % (len_, count))

    print("***************** Target Length Statistics ******************")
    len_counter = {}
    for src_tokens, trgs_tokens in tokenized_train_pairs:
        for trgs_token in trgs_tokens:
            len_count = len_counter.get(len(trgs_token), 0) + 1
            len_counter[len(trgs_token)] = len_count

    sorted_len = sorted(len_counter.items(), key=lambda x:x[0], reverse=True)

    for len_, count in sorted_len:
        print('%d,%d' % (len_, count))

    '''
    dump to disk
    '''
    print("Dumping dict to disk: %s" % config['save_data'] + '.vocab.pt')
    torch.save([word2id, id2word], open(config['save_data'] + '.vocab.pt', 'wb'))
    print("Dumping valid to disk: %s" % (config['save_data'] + '.valid_dump.pt'))
    torch.save(valid_dump, open(config['save_data'] + '.valid_dump.pt', 'wb'))
    print("Dumping test to disk: %s" % (config['save_data'] + '.test_dump.pt'))
    torch.save(test_dump, open(config['save_data'] + '.test_dump.pt', 'wb'))
    print("Dumping done!")

if __name__ == "__main__":
    main()
