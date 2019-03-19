# -*- coding: utf-8 -*-
"""
Remove noisy items (abstract contains "Full textFull text is available as a scanned copy of the original print version.") (around 132561 out of 3114539=2981978) and remove duplicates by title
"""
import json

from pykp.data.remove_duplicates import example_iterator_from_json

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    mag_path = "source_data/mag_output/mag_nodup.json"
    mag_output_path = "source_data/mag_output/mag_nodup_plus.json"
    kp20k_train_path = "source_data/kp20k/kp20k_training.json"

    train_dataset_name = 'mag'
    test_dataset_names = ['kp20k_train']
    id_field = 'id'
    title_field = 'title'
    text_field = 'abstract'
    keyword_field = 'keywords'
    trg_delimiter = None

    mag_examples_iter = list(example_iterator_from_json(path=mag_path,
                                                     dataset_name="mag",
                                                     id_field=id_field,
                                                     title_field=title_field,
                                                     text_field=text_field,
                                                     keyword_field=keyword_field,
                                                     trg_delimiter=trg_delimiter))
    print("Loaded %d examples from MAG" % len(mag_examples_iter))

    id_field = None
    keyword_field = 'keywords'
    trg_delimiter = ';'
    kp20k_train_examples = list(example_iterator_from_json(path=kp20k_train_path,
                                                    dataset_name="kp20k_train",
                                                    id_field=id_field,
                                                    title_field=title_field,
                                                    text_field=text_field,
                                                    keyword_field=keyword_field,
                                                    trg_delimiter=trg_delimiter))

    print("Loaded %d examples from KP20k train" % len(kp20k_train_examples))

    title_pool = set()
    for ex in kp20k_train_examples:
        title_pool.add(ex["title"].lower().strip())

    non_dup_count = 0
    with open(mag_output_path, 'w') as mag_output:
        for ex_id, ex in enumerate(mag_examples_iter):
            title = ex["title"].lower().strip()
            if title not in title_pool:
                non_dup_count += 1
                title_pool.add(title)
                mag_output.write(json.dumps(ex) + '\n')
                if ex_id % 1000 == 0:
                    print("non-dup/processed/all = %d/%d/%d" % (non_dup_count, ex_id, len(mag_examples_iter)))

