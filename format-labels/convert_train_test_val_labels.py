import os
import argparse

def create_dict(train_files, test_files, val_files):
    # Parse each of the txt files and cache to dict
    dict = {}

    for t in train_files:
        with open(t, 'rb') as f:
            for line in f:
                split_line = line.split()[0].decode("utf-8")
                dict[split_line] = "train"

    for t in test_files:
        with open(t, 'rb') as f:
            for line in f:
                split_line = line.split()[0].decode("utf-8")
                dict[split_line] = "test"

    for t in val_files:
        with open(t, 'rb') as f:
            for line in f:
                split_line = line.split()[0].decode("utf-8")
                dict[split_line] = "val"

    # print("dictionary length: ", len(dict))
    return dict