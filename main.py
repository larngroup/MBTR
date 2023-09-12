import tensorflow as tf
import numpy as np
import pandas as pd
from bert_smiles import *
from arg_parse import *
import itertools
from train_util import *
from positionals import *
from dataset_builder_util import *
from output_layer import *
from itertools import combinations_with_replacement
from layers_utils import *


def build_model(FLAGS, prot_emb_size, prot_atv_fun, prot_enc_layers, prot_heads, prot_dff, drop_rate,
                   smiles_emb_size, smiles_atv_fun, smiles_enc_layers, smiles_heads, smiles_dff, fc_layers, fc_units,
                   fc_activation):

    """
    Function to build the Model

    Args:
    - FLAGS: arguments object
    - prot_emb_size [int] : protein embedding size
    - prot_atv_fun: Protein Trasnformer-Encoder dense layers activation function
    - prot_enc_layers [int]: number of Protein Transformer-Encoders
    - prot_heads [int]: number of heads for the Protein MHSA
    - prot_dff [int]: hidden numbers for the first dense layer of the Protein PWFFN
    - drop_rate [float]: % of dropout
    - smiles_emb_size [int] : protein embedding size
    - smiles_atv_fun: SMILES Trasnformer-Encoder dense layers activation function
    - smiles_enc_layers [int]: number of SMILES Transformer-Encoders
    - smiles_heads [int]: number of heads for the SMILES MHSA
    - smiles_dff [int]: hidden numbers for the first dense layer of the SMILES PWFFN
    - fc_layers [int]: PWMLP number of layers
    - fc_units [list of ints]: hidden neurons for each one of the dense layers of the PWMLP
    - fc_activation: PWMLP activation function

    Outputs:
    - Model
    """

    prot_atv_fun = af_config(prot_atv_fun)
    smiles_atv_fun= af_config(smiles_atv_fun)
    fc_activation = af_config(fc_activation)

    # Model Input
    prot_input = tf.keras.Input(FLAGS.prot_len, dtype=tf.int64, name='protein_input')
    smiles_input = tf.keras.Input(FLAGS.smiles_len + 1, dtype=tf.int64, name='smiles_input')

    # Protein & SMILES Mask
    prot_pad_mask = attn_pad_mask()(prot_input)
    smiles_pad_mask = attn_pad_mask()(smiles_input)

    # Protein & SMILES Embedding
    prot_emb = tf.keras.layers.Embedding(FLAGS.prot_dict_len + 3, prot_emb_size, name='protein_embedding')(prot_input)
    smiles_emb = tf.keras.layers.Embedding(FLAGS.smiles_dict_len + 3, smiles_emb_size, name='smiles_embedding')(smiles_input)

    # SMILES Positional Encoding
    char_embedding = PosLayer(FLAGS.smiles_len + 1, smiles_emb_size, drop_rate, name='smiles_pos_layer')(smiles_emb)

    # SMILES Transformer-Encoder
    smiles_tokens, _ = Encoder(smiles_emb_size, smiles_enc_layers, smiles_heads, smiles_dff, smiles_atv_fun, drop_rate,
                               FLAGS.smiles_dim_k, FLAGS.smiles_param_share,
                               FLAGS.smiles_full_attention, FLAGS.smiles_return_intermediate,
                               name='smiles_trans_encoder')(char_embedding, smiles_pad_mask)

    # Binding Vector Prediction
    if smiles_tokens.shape[-1] != prot_emb_size:
        smiles_cls_token = BertPooler(prot_emb_size, smiles_atv_fun, 1, drop_rate, name='smiles_pooler')(smiles_tokens)

    else:
        smiles_cls_token = tf.expand_dims(tf.gather(smiles_tokens, 0, axis=1), axis=1)

    combined_info = tf.concat([smiles_cls_token, prot_emb], axis=1)

    pad_mask = tf.concat([tf.expand_dims(tf.gather(prot_pad_mask, 0, axis=-1) * 0, axis=-1), prot_pad_mask], axis=-1)

    cond_positions = [1] + [2] * FLAGS.prot_len

    combined_info = PositionalsLayer(2, FLAGS.prot_len + 1, prot_emb_size, drop_rate)(combined_info, cond_positions)

    combined_tokens, _ = Encoder(prot_emb_size, prot_enc_layers, prot_heads,
                               prot_dff, prot_atv_fun, drop_rate, FLAGS.prot_dim_k, FLAGS.prot_param_share,
                               FLAGS.prot_full_attention, FLAGS.prot_return_intermediate,
                               name='prot_trans_encoder')(combined_info, pad_mask)

    outs = OutputMLP(fc_layers, fc_units, fc_activation, FLAGS.output_act_fun, drop_rate,
                     name='output_block')(combined_tokens)

    return tf.keras.Model(inputs=[prot_input, smiles_input], outputs=outs)

def load_train_model(FLAGS, params=[256,'gelu',2,8,1024,0.1,256,'gelu',2,8,1024,3,[128,64,32],'gelu'],
                     opt=['radam', '5e-04', '0.9', '0.999', '1e-08', '1e-05', '136100', '0.02', '1e-05'],
                     checkpoint_path='./model_checkpoint/'):

    mbtr_model = build_model(FLAGS, *params)

    model_opt = opt_config(opt)

    global_var = tf.Variable(1)
    checkpoint_object = tf.train.Checkpoint(step=global_var, optimizer=model_opt,
                                            model=mbtr_model)
    latest = tf.train.latest_checkpoint(checkpoint_path)
    checkpoint_object.restore(latest).expect_partial()
    mbtr_model = checkpoint_object.model

    return mbtr_model

def grid_search(FLAGS, data_train, data_val, data_test, model_function):
    """
    Grid Search function

    Args:
    - FLAGS: arguments object
    - dataset_train [TF Dataset]: [train_protein_data, train_smiles_data, train_target, train_weights]
    - dataset_val [TF Dataset]: [val_protein_data, val_smiles_data, val_target, val_weights]
    - dataset_test [TF Dataset]: [test_protein_data, test_smiles_data, test_target, test_weights]
    - model_function: function that creates the model
    """

    prot_emb_list = FLAGS.prot_emb_size
    prot_af_list = FLAGS.prot_atv_fun
    prot_enc_depth_list = FLAGS.prot_enc_layers
    prot_enc_heads_list = FLAGS.prot_enc_heads
    prot_dff_list = FLAGS.prot_enc_dff
    drop_rate_list = FLAGS.drop_rate
    smiles_emb_list = FLAGS.smiles_emb_size
    smiles_af_list = FLAGS.smiles_atv_fun
    smiles_enc_depth_list = FLAGS.smiles_enc_layers
    smiles_enc_heads_list = FLAGS.smiles_enc_heads
    smiles_dff_list = FLAGS.smiles_enc_dff

    fc_layers_units_list = []
    for i in FLAGS.num_fc_layer:
        for k in combinations_with_replacement(FLAGS.num_units_fc_layers[0], i):
            fc_layers_units_list.append([i, list(k)])

    # fc_layers_units_list = []
    # for i in FLAGS.num_fc_layer:
    #     for j in FLAGS.num_units_fc_layers:
    #         fc_layers_units_list.append([i, list(j)])

    # fc_layers_list = FLAGS.num_fc_layer
    # fc_units_list = FLAGS.num_units_fc_layers
    fc_activation_list = FLAGS.fc_act_fun
    batch_size_list = FLAGS.batch_size
    epochs_list = FLAGS.epochs
    loss_option_list = FLAGS.loss_opt
    optimizer_list = FLAGS.optimizer_fn

    # for params in itertools.product(prot_emb_list, prot_af_list, prot_enc_depth_list, prot_enc_heads_list,
    #                                 prot_dff_list, drop_rate_list, smiles_emb_list, smiles_af_list,
    #                                 smiles_enc_depth_list, smiles_enc_heads_list, smiles_dff_list,
    #                                 fc_layers_list, fc_units_list, fc_activation_list, batch_size_list,
    #                                 epochs_list, loss_option_list, optimizer_list):

    for params in itertools.product(prot_emb_list, prot_af_list, prot_enc_depth_list, prot_enc_heads_list,
                                    prot_dff_list, drop_rate_list, smiles_emb_list, smiles_af_list,
                                    smiles_enc_depth_list, smiles_enc_heads_list, smiles_dff_list,
                                    fc_layers_units_list, fc_activation_list, batch_size_list,
                                    epochs_list, loss_option_list, optimizer_list):


        FLAGS.log_dir = os.getcwd() + '/logs/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"
        FLAGS.checkpoint_dir = os.getcwd() + '/checkpoints/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"

        if not os.path.exists(FLAGS.log_dir):
            os.makedirs(FLAGS.log_dir)
        if not os.path.exists(FLAGS.checkpoint_dir):
            os.makedirs(FLAGS.checkpoint_dir)

        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17 = params


        logging(FLAGS, str(FLAGS))
        logging(FLAGS, "--------------------Grid Search-------------------")

        # Model Optimizer
        p17_v2 = opt_config(p17)

        logging(FLAGS,
                (
                            "Prot Emb Size: %d, Prot AF: %s, Prot Enc Depth: %d, Prot Heads: %d, Prot DFF: %d, " +
                            "Dropout Rate: %0.4f, SMILES Emb Size: %d, " +
                            "SMILES AF: %s, SMILES Enc Depth: %d, SMILES Heads: %d, SMILES DFF: %d, FC Depth: %d, " +
                            "FC Units: %s, " +
                            "FC AF: %s, Batch Size: %d, Epochs: %d, Loss Option: %s, Optimizer: %s ") %
                (p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12[0], p12[-1], p13, p14, p15, p16, p17_v2.get_config()))

        # Training-Validation (Hold-out) + Testing
        data_train_batch = data_train.batch(p14)
        data_val_batch = data_val.batch(p14)
        data_test_batch = data_test.batch(p14)

        # Model
        model = model_function(FLAGS, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12[0], p12[-1], p13)

        # Final Dense Layer Initial Bias (Imbalanced Data Correction) np.log(pos/neg)
        initial_bias = np.array([-3.6595278])
        model.get_layer('output_block').get_layer('out_final').set_weights(
            [model.get_layer('output_block').get_layer('out_final').get_weights()[0]] + [initial_bias])

        # Model Summary
        model.summary()

        # Checkpoint Object and Checkpoint Manager
        global_var = tf.Variable(1)

        checkpoint_object = tf.train.Checkpoint(step=global_var, optimizer=p17_v2, model=model)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint_object, directory=FLAGS.checkpoint_dir,
                                                        max_to_keep=1)

        # Loss Function
        if p16[0] == 'focal':
            loss_fn = tf.keras.losses.BinaryFocalCrossentropy(from_logits=True,
                                                                       gamma=float(p16[1]),
                                                                       reduction=tf.keras.losses.Reduction.NONE)

        elif p16[0] == 'standard':
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                         reduction=tf.keras.losses.Reduction.NONE)
        # Training Objects
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.Mean(name='train_acc')
        train_f1 = tf.keras.metrics.Mean(name='train_f1')
        train_recall = tf.keras.metrics.Mean(name='train_recall')
        train_precision = tf.keras.metrics.Mean(name='train_precision')
        train_mcc = tf.keras.metrics.Mean(name='train_mcc')

        # Validation Objects
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_accuracy = tf.keras.metrics.Mean(name='val_acc')
        val_f1 = tf.keras.metrics.Mean(name='val_f1')
        val_recall = tf.keras.metrics.Mean(name='val_recall')
        val_precision = tf.keras.metrics.Mean('val_precision')
        val_mcc = tf.keras.metrics.Mean(name='val_mcc')

        # Testing Objects
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.Mean(name='test_acc')
        test_f1 = tf.keras.metrics.Mean(name='test_f1')
        test_recall = tf.keras.metrics.Mean(name='test_recall')
        test_precision = tf.keras.metrics.Mean('test_precision')
        test_mcc = tf.keras.metrics.Mean(name='test_mcc')

        # Run Grid Search: Hold-out Validation
        run_train_val(FLAGS, p15, data_train_batch, data_val_batch, data_test_batch, model, p17_v2,
                      loss_fn, p16, train_loss, train_accuracy, train_f1, train_recall, train_precision, train_mcc,
                      val_loss, val_accuracy, val_f1, val_recall, val_precision, val_mcc,
                      test_loss, test_accuracy, test_f1, test_recall, test_precision, test_mcc,
                      checkpoint_object, checkpoint_manager)



def run_grid_search(FLAGS):
    """
    Run Grid Search function
    Args:
    - FLAGS: arguments object
    """

    dataset_train, dataset_val = shuffle_split(load_data('./data/metals.dataset/prot.tfrecords',
                                                         './data/metals.dataset/smiles.tfrecords',
                                                         './data/metals.dataset/target.tfrecords',
                                                         './data/metals.dataset/weights.tfrecords'), 0.9)

    dataset_test = load_data('./data/zn.dataset/prot.tfrecords',
                             './data/zn.dataset/smiles.tfrecords',
                             './data/zn.dataset/target.tfrecords',
                             './data/zn.dataset/weights.tfrecords')
    model_fn = build_model
    grid_search(FLAGS, dataset_train, dataset_val, dataset_test, model_fn)


def run_train_mode(FLAGS):
    dataset_train, dataset_val = shuffle_split(load_data('./data/metals.dataset/prot.tfrecords',
                                                         './data/metals.dataset/smiles.tfrecords',
                                                         './data/metals.dataset/target.tfrecords',
                                                         './data/metals.dataset/weights.tfrecords'), 0.9)

    dataset_test = load_data('./data/zn.dataset/prot.tfrecords',
                             './data/zn.dataset/smiles.tfrecords',
                             './data/zn.dataset/target.tfrecords',
                             './data/zn.dataset/weights.tfrecords')

    data_train_batch = dataset_train.batch(FLAGS.batch_size[0]).take(1)
    data_val_batch = dataset_val.batch(FLAGS.batch_size[0]).take(1)
    data_test_batch = dataset_test.batch(FLAGS.batch_size[0]).take(1)

    FLAGS.log_dir = os.getcwd() + '/logs/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"
    FLAGS.checkpoint_dir = os.getcwd() + '/checkpoints/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    logging(FLAGS, str(FLAGS))
    logging(FLAGS, "--------------------Training/Validation-------------------")


    model_opt = opt_config(FLAGS.optimizer_fn[0])

    loss_opt = FLAGS.loss_opt[0]

    if loss_opt[0] == 'focal':
        loss_fn = tf.keras.losses.BinaryFocalCrossentropy(from_logits=True,
                                                          gamma=float(loss_opt[1]),
                                                          reduction=tf.keras.losses.Reduction.NONE)

    elif loss_opt[0] == 'standard':
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                     reduction=tf.keras.losses.Reduction.NONE)


    logging(FLAGS,
            (
                    "Prot Emb Size: %d, Prot AF: %s, Prot Enc Depth: %d, Prot Heads: %d, Prot DFF: %d, " +
                    "Dropout Rate: %0.4f, SMILES Emb Size: %d, " +
                    "SMILES AF: %s, SMILES Enc Depth: %d, SMILES Heads: %d, SMILES DFF: %d, FC Depth: %d, " +
                    "FC Units: %s, " +
                    "FC AF: %s, Batch Size: %d, Epochs: %d, Loss Option: %s, Optimizer: %s ") %
                    (FLAGS.prot_emb_size[0], FLAGS.prot_atv_fun[0], FLAGS.prot_enc_layers[0],
                    FLAGS.prot_enc_heads[0], FLAGS.prot_enc_dff[0], FLAGS.drop_rate[0],
                    FLAGS.smiles_emb_size[0], FLAGS.smiles_atv_fun[0], FLAGS.smiles_enc_layers[0],
                    FLAGS.smiles_enc_heads[0], FLAGS.smiles_enc_dff[0],
                    FLAGS.num_fc_layer[0], FLAGS.num_units_fc_layers[0],
                    FLAGS.fc_act_fun[0], FLAGS.batch_size[0], FLAGS.epochs[0], loss_opt, model_opt.get_config()))




    model = build_model(FLAGS, FLAGS.prot_emb_size[0], FLAGS.prot_atv_fun[0], FLAGS.prot_enc_layers[0],
                           FLAGS.prot_enc_heads[0], FLAGS.prot_enc_dff[0], FLAGS.drop_rate[0],
                           FLAGS.smiles_emb_size[0], FLAGS.smiles_atv_fun[0], FLAGS.smiles_enc_layers[0],
                           FLAGS.smiles_enc_heads[0], FLAGS.smiles_enc_dff[0],
                           FLAGS.num_fc_layer[0], FLAGS.num_units_fc_layers[0],
                           FLAGS.fc_act_fun[0])


    initial_bias = np.array([-3.6595278])
    model.get_layer('output_block').get_layer('out_final').set_weights(
        [model.get_layer('output_block').get_layer('out_final').get_weights()[0]] + [initial_bias])

    # Model Summary
    model.summary()

    # Checkpoint Object and Checkpoint Manager
    global_var = tf.Variable(1)

    checkpoint_object = tf.train.Checkpoint(step=global_var, optimizer=model_opt, model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint_object, directory=FLAGS.checkpoint_dir,
                                                    max_to_keep=1)

    # Training Objects
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_acc')
    train_f1 = tf.keras.metrics.Mean(name='train_f1')
    train_recall = tf.keras.metrics.Mean(name='train_recall')
    train_precision = tf.keras.metrics.Mean(name='train_precision')
    train_mcc = tf.keras.metrics.Mean(name='train_mcc')

    # Validation Objects
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.Mean(name='val_acc')
    val_f1 = tf.keras.metrics.Mean(name='val_f1')
    val_recall = tf.keras.metrics.Mean(name='val_recall')
    val_precision = tf.keras.metrics.Mean('val_precision')
    val_mcc = tf.keras.metrics.Mean(name='val_mcc')

    # Testing Objects
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.Mean(name='test_acc')
    test_f1 = tf.keras.metrics.Mean(name='test_f1')
    test_recall = tf.keras.metrics.Mean(name='test_recall')
    test_precision = tf.keras.metrics.Mean('test_precision')
    test_mcc = tf.keras.metrics.Mean(name='test_mcc')


    run_train_val(FLAGS, FLAGS.epochs[0], data_train_batch, data_val_batch, data_test_batch, model, model_opt,
                  loss_fn, loss_opt, train_loss, train_accuracy, train_f1, train_recall, train_precision, train_mcc,
                  val_loss, val_accuracy, val_f1, val_recall, val_precision, val_mcc,
                  test_loss, test_accuracy, test_f1, test_recall, test_precision, test_mcc,
                  checkpoint_object, checkpoint_manager)


def run_eval_mode(FLAGS):
    dataset_test = load_data('./data/zn.dataset/prot.tfrecords',
                             './data/zn.dataset/smiles.tfrecords',
                             './data/zn.dataset/target.tfrecords',
                             './data/zn.dataset/weights.tfrecords')

    dataset_test = [i for i in dataset_test.as_numpy_iterator()]
    prot_data = tf.stack([i[0] for i in dataset_test], axis=0)
    smiles_data = tf.stack([i[1] for i in dataset_test], axis=0)
    target = tf.stack([i[2] for i in dataset_test], axis=0)
    weights = tf.stack([i[3] for i in dataset_test], axis=0)

    model = load_train_model(FLAGS)

    preds = model([prot_data, smiles_data], training=False)

    bind_metrics = 'Balanced Accuracy: %0.4f, Recall: %0.4f, Precision: %0.4f, F1-Score: %0.4f, MCC: %0.4f' % \
                       metrics_function(target, preds, weights)

    print('-----------Testing Set Metrics-----------')
    print(bind_metrics)



if __name__ == '__main__':
    FLAGS = arg_parser()

    if FLAGS.inference_option == 'Train':
        run_train_mode(FLAGS)

    if FLAGS.inference_option == 'Validation':
        run_grid_search(FLAGS)

    if FLAGS.inference_option == 'Evaluation':
        run_eval_mode(FLAGS)

