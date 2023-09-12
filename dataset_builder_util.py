# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""

import pandas as pd
import numpy as np
import random
from layers_utils import *
import json
import re
from operator import itemgetter
import periodictable as pt
import codecs
from subword_nmt.apply_bpe import BPE


class dataset_builder():
    """
    Set of Dataset processing functions

    Args:
    - data_path [Dict]: Dataset Path, Prot BPE Codes Path,
                        Prot BPE Codes Map Path, SMILES Dictionary Path,

    """

    def __init__(self, data_path, **kwargs):
        super(dataset_builder, self).__init__(**kwargs)
        self.data_path = data_path

    def get_data(self):
        dataset = ""
        if self.data_path['data'] is not None:
            dataset = pd.read_csv(self.data_path['data'], sep='\t', memory_map=True)
        codes = ""
        if self.data_path['bpe_codes'] is not None:
            codes = codecs.open(self.data_path['bpe_codes'])
        codes_map = ""
        if self.data_path['bpe_codes_map'] is not None:
            codes_map = pd.read_csv(self.data_path['bpe_codes_map'])
        smiles_dictionary = ""
        if self.data_path['smiles_dict'] is not None:
            smiles_dictionary = json.load(open(self.data_path['smiles_dict']))

        return dataset, smiles_dictionary, codes, codes_map

    # Convert Tokens to Integer Values using the Dictionary and padded up to a maximum value of max_len
    def smiles_data_conversion(self, data, dictionary, max_len):
        keys = list(i for i in dictionary.keys() if len(i) > 1)

        if len(keys) == 0:
            data = pd.DataFrame([list(i) for i in data])

        else:
            char_list = []
            for i in data:
                positions = []
                for j in keys:
                    positions.extend([(k.start(), k.end() - k.start()) for k in re.finditer(j, i)])

                positions = sorted(positions, key=itemgetter(0))

                if len(positions) == 0:
                    char_list.append(list(i))

                else:
                    new_list = []
                    j = 0
                    positions_start = [k[0] for k in positions]
                    positions_len = [k[1] for k in positions]

                    while j < len(i):
                        if j in positions_start:
                            new_list.append(str(i[j] + i[j + positions_len[positions_start.index(j)] - 1]))
                            j = j + positions_len[positions_start.index(j)]
                        else:
                            new_list.append(i[j])
                            j = j + 1
                    char_list.append(new_list)

            data = pd.DataFrame(char_list)

        data.replace(dictionary, inplace=True)

        data = data.fillna(0)
        if len(data.iloc[0, :]) == max_len:
            return data
        elif len(data.iloc[0, :]) < max_len:
            zeros_array = np.zeros(shape=(len(data.iloc[:, 0]), max_len - len(data.iloc[0, :])))
            data = pd.concat((data, pd.DataFrame(zeros_array)), axis=1)
            return data
        elif len(data.iloc[0, :]) > max_len:
            data = data.iloc[:, :max_len]
            return data

    # Protein FCS/BPE Encoding
    def encoding_bpe(self, data, codes, codes_map, max_len):
        bpe = BPE(codes, merges=-1, separator='')
        idx2word = codes_map['index'].values
        idx2word[7037] = 'NA'
        words2idx = dict(zip(idx2word, range(0, len(idx2word))))

        vectors = []
        idx_list = []

        for k in data:
            t1 = bpe.process_line(k).split()  # split

            t2 = []
            for i in range(len(t1)):
                if i == 0:
                    t2.append([j for j in range(len(t1[i]))])
                else:
                    t2.append([j + t2[i - 1][-1] + 1 for j in range(len(t1[i]))])

            try:
                i1 = np.asarray([words2idx[j] + 1 for j in t1])  # index
            except:
                i1 = np.array([0])

            l = len(i1)

            if l < max_len:
                k = np.pad(i1, (0, max_len - l), 'constant', constant_values=0)
            else:
                k = i1[:max_len]
            vectors.append(k[None, :])
            idx_list.append(t2)

        return tf.cast(tf.concat(vectors, axis=0), dtype=tf.int32), idx_list

    # Convert to Tensor Format
    def transform_smiles(self, smiles_column, smiles_max_len):

        smiles_data = self.smiles_data_conversion(self.get_data()[0][smiles_column],
                                                  self.get_data()[1], smiles_max_len).astype('int64')

        return tf.convert_to_tensor(smiles_data, dtype=tf.int32)

    # Convert to Tensor Format
    def transform_proteins(self, protein_column, prot_max_len):

        prot_data = \
            self.encoding_bpe(self.get_data()[0][protein_column], self.get_data()[2], self.get_data()[3], prot_max_len)[
                0]

        return tf.convert_to_tensor(prot_data, dtype=tf.int32)

def shuffle_split(dataset, train_perc):
    """
    Hold-Out Training/Validation Split - Metal Binding Dataset

    Args:
    - dataset [TF Dataset]: dataset in TF Dataset format
    - train_perc [int]: % Validation Ratio
    """

    dataset = dataset.shuffle(len(dataset), seed=12345, reshuffle_each_iteration=False)
    dataset_train = dataset.take(int(train_perc * len(dataset)))
    dataset_val = dataset.skip(int(train_perc * len(dataset)))

    return dataset_train, dataset_val


#----------- Binding Dataset TFRecords Load Functions-----------
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))


def _serialize(prot, smiles, target, weights):
    feature_prot = {
        'prot': _bytes_feature(prot)
    }
    feature_smiles = {
        'smiles': _bytes_feature(smiles)
    }

    feature_target = {
        'target': _bytes_feature(target)
    }

    feature_weights = {
        'weights': _bytes_feature(weights)
    }

    prot_example = tf.train.Example(features=tf.train.Features(feature=feature_prot))
    smiles_example = tf.train.Example(features=tf.train.Features(feature=feature_smiles))
    target_example = tf.train.Example(features=tf.train.Features(feature=feature_target))
    weights_example = tf.train.Example(features=tf.train.Features(feature=feature_weights))

    return prot_example.SerializeToString(), smiles_example.SerializeToString(), target_example.SerializeToString(), \
        weights_example.SerializeToString()


def _process(prot, smiles, target, weights):
    serialized_prot = tf.io.serialize_tensor(tf.convert_to_tensor(prot, dtype=tf.int32))
    serialized_smiles = tf.io.serialize_tensor(tf.convert_to_tensor(smiles, dtype=tf.int32))
    serialized_target = tf.io.serialize_tensor(tf.convert_to_tensor(target, dtype=tf.int32))
    serialized_weights = tf.io.serialize_tensor(tf.convert_to_tensor(weights, dtype=tf.float32))

    return _serialize(serialized_prot, serialized_smiles, serialized_target, serialized_weights)


def parse_prot(data_example):
    feature_description = {
        'prot': tf.io.FixedLenFeature([], tf.string)
    }

    parsed_data = tf.io.parse_single_example(
        data_example, feature_description)

    data = tf.io.parse_tensor(parsed_data['prot'], tf.int32)

    return data


def parse_smiles(data_example):
    feature_description = {
        'smiles': tf.io.FixedLenFeature([], tf.string)
    }

    parsed_data = tf.io.parse_single_example(
        data_example, feature_description)

    data = tf.io.parse_tensor(parsed_data['smiles'], tf.int32)

    return data


def parse_target(data_example):
    feature_description = {
        'target': tf.io.FixedLenFeature([], tf.string)
    }

    parsed_data = tf.io.parse_single_example(
        data_example, feature_description)
    data = tf.io.parse_tensor(parsed_data['target'], tf.int32)

    return data


def parse_weights(data_example):
    feature_description = {
        'weights': tf.io.FixedLenFeature([], tf.string)
    }

    parsed_data = tf.io.parse_single_example(
        data_example, feature_description)

    data = tf.io.parse_tensor(parsed_data['weights'], tf.float32)

    return data


def load_data(path_prot='./data/prot.tfrecords', path_smiles='./data/smiles.tfrecords',
              path_target='./data/target.tfrecords', path_weights='./data/weights.tfrecords'):
    prot = tf.data.TFRecordDataset(path_prot)
    prot = prot.map(parse_prot)
    prot = [i.numpy() for i in prot][0]

    smiles = tf.data.TFRecordDataset(path_smiles)
    smiles = smiles.map(parse_smiles)
    smiles = [i.numpy() for i in smiles][0]

    target = tf.data.TFRecordDataset(path_target)
    target = target.map(parse_target)
    target = [i.numpy() for i in target][0]

    weights = tf.data.TFRecordDataset(path_weights)
    weights = weights.map(parse_weights)
    weights = [i.numpy() for i in weights][0]

    dataset = tf.data.Dataset.from_tensor_slices((prot, smiles, target, weights))

    return dataset
