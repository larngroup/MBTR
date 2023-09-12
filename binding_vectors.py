from urllib.request import urlopen
import urllib

import pandas as pd

from main import *


def get_uniprot_seq(id):
    try:
        uniprot_url = "https://www.uniprot.org/uniprot/{0}.fasta"
        uni_prot = urllib.request.urlopen(uniprot_url.format(id))
        uni_prot = uni_prot.readlines()
        uni_prot = "".join([line.decode("utf-8").rstrip() for line in uni_prot[1:]])

        return uni_prot

    except Exception as e:
        print('Uniprot Fail')
        return []


def get_binding_vector(FLAGS, prot_ids, prot_seqs, smiles_list, file_name=None):
    prot_seqs_list = []
    if prot_ids is not None:
        for i in prot_ids:
            prot_seqs_list += [get_uniprot_seq(i)]

    if prot_seqs is not None:
        prot_seqs_list.extend(prot_seqs)

    data_path = {'data': None, 'bpe_codes': './dictionary/protein_codes_uniprot.txt',
                 'bpe_codes_map': './dictionary/subword_units_map_uniprot.csv',
                 'smiles_dict': './dictionary/metal_dict.txt'}

    _, smiles_dict, codes, codes_map = dataset_builder(data_path).get_data()

    prot_dict = {i + 1: j for i, j in zip(codes_map.iloc[:, 1], codes_map.iloc[:, 2])}

    prot_sequences = tf.convert_to_tensor(dataset_builder(data_path).
                                          encoding_bpe(prot_seqs_list,
                                                       codes, codes_map, FLAGS.prot_len)[0],
                                          dtype=tf.int32)

    smiles_strings = add_reg_token(tf.convert_to_tensor(dataset_builder(data_path).
                                                        smiles_data_conversion(pd.Series(smiles_list), smiles_dict,
                                                                               FLAGS.smiles_len),
                                                        dtype=tf.int32),
                                   FLAGS.smiles_dict_len)

    mbtr_model = load_train_model(FLAGS)
    preds = tf.cast(tf.nn.sigmoid(mbtr_model([prot_sequences, smiles_strings], training=False)) > 0.5, tf.int64)[:, :,
            0]
    preds = preds.numpy()
    prot_sequeces = prot_sequences.numpy()

    results = []
    for i in range(len(prot_sequeces)):
        results.append(
            [[str(j) for j in pd.DataFrame(prot_sequeces[i][prot_sequeces[i] != 0]).replace(prot_dict).values],
             [str(j) for j in preds[i][prot_sequeces[i] != 0]],
             [str(j) for j in
              pd.Series(preds[i][prot_sequeces[i] != 0]).loc[preds[i][prot_sequeces[i] != 0] != 0].index.to_list()],
             [str(j) for j in pd.Series(prot_sequences[i]).loc[tf.math.logical_and(prot_sequeces[i] != 0,
                                                                                   preds[i] != 0).numpy()].replace(
                 prot_dict).to_list()]])

    for i in range(len(results)):
        print("Protein Tokens: %s,\n Protein Binding Vector: %s,\n"
              " Protein Binding Positions: %s,\n"
              " Protein Binding Residues: %s\n" % (str(results[i][0]), str(results[i][1]),
                                                   str(results[i][2]), str(results[i][3])))

    if file_name is None:
        file_name = "results_{}".format(time.strftime("%d_%m_%y_%H_%M", time.gmtime()))

    if not os.path.exists(FLAGS.bind_results_path):
        os.mkdir(FLAGS.bind_results_path)

    with open(FLAGS.bind_results_path + '/' + file_name + '.txt', 'w') as fw:
        for i in range(len(results)):
            if i <= len(prot_ids)-1:
                fw.write('%s\n' % prot_ids[i])
            if i <= len(prot_seqs_list)-1:
                fw.write('%s\n' % prot_seqs_list[i])
            fw.write('%s\n' % ' '.join(results[i][0]))
            fw.write('%s\n' % ' '.join(results[i][1]))
            fw.write('%s\n' % ' '.join(results[i][2]))
            fw.write('%s\n' % ' '.join(results[i][3]))

    return results


# uniprot_ids_set1 = ['A9CJ36', 'A9QSF3', 'X5K3J9', 'P0ACL7', 'P39161', 'P0ACL2']
# uniprot_ids_set2 = ['P0A8W0', 'P31460', 'A0A4Y3TY52', 'Q19AK4']
# uniprot_ids_set3 = ['Q9WYS0']
# uniprot_ids_set4 = ['Q8NLM6']
# uniprot_ids_set5 = ['P0A8V6', 'A0A0H2UKZ1', 'A0A0H0Y8X2', 'Q0SB06', 'Q46SA5', 'Q8Y982']
# uniprot_ids_set6 = ['P10585']
#
# uniprot_ids_set = uniprot_ids_set1 + uniprot_ids_set2 + uniprot_ids_set3 + uniprot_ids_set4 + uniprot_ids_set5 + uniprot_ids_set6
#
# results = get_binding_vector(FLAGS, uniprot_ids_set, None, ['[Zn+2]'] * len(uniprot_ids_set))
# with open('./results/results_set_sigmoid05_v3.txt', 'w') as fw:
#     for i in range(len(results)):
#         fw.write('%s\n' % uniprot_ids_set[i])
#         fw.write('%s\n' % ' '.join(results[i][0]))
#         fw.write('%s\n' % ' '.join(results[i][1]))
#         fw.write('%s\n' % ' '.join(results[i][2]))
#         fw.write('%s\n' % ' '.join(results[i][3]))

if __name__ == '__main__':
    FLAGS = arg_parser()

    get_binding_vector(FLAGS, FLAGS.prot_id_list_bind_vector,
                       FLAGS.prot_seq_list_bind_vector, FLAGS.smiles_list_bind_vector)
