import tensorflow as tf
import time
from arg_parse import *

# Weighted Binary Cross Entropy or Binary Focal Cross Entropy loss value
def loss_function(loss_object, real, preds, weights, loss_opt):
    ones = tf.cast(tf.math.logical_and(tf.math.equal(real, 1), tf.equal(weights, 1)), tf.float32)
    zeros = tf.cast(tf.math.logical_and(tf.math.equal(real, 0), tf.equal(weights, 1)), tf.float32)

    score_ones = tf.reduce_sum(ones) / (tf.reduce_sum(ones) + tf.reduce_sum(zeros))

    if loss_opt[0] == 'focal' and loss_opt[2] == '' and loss_opt[3] == '':
        weights = ones + (1 - score_ones) + zeros * score_ones

    elif loss_opt[0] == 'focal':
        weights = zeros * float(loss_opt[2]) + ones * float(loss_opt[3])

    elif loss_opt[0] == 'standard' and loss_opt[1] == '' and loss_opt[2] == '':
        weights = ones * (1 - score_ones) + zeros * score_ones

    elif loss_opt[0] == 'standard':
        weights = zeros * float(loss_opt[1]) + ones * float(loss_opt[2])

    # weights = ones * 0.60 + zeros * 0.40

    loss_value = loss_object(real, preds, weights)

    loss_value = tf.reduce_sum(loss_value) / tf.reduce_sum(weights)

    return loss_value

# Evaluation metrics
def metrics_function(real, preds, weights, threshold=0.5):
    real = tf.cast(real, dtype=tf.int64)
    preds = tf.cast(tf.nn.sigmoid(preds) > threshold, tf.int64)
    # preds = tf.cast(tf.argmax(preds, axis=2), dtype=tf.int64)
    weights = tf.math.logical_not(tf.math.equal(weights, 0))

    tp = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_and(tf.equal(real, 1), tf.equal(preds, 1)), weights),
                               dtype=tf.float32))
    tn = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_and(tf.equal(real, 0), tf.equal(preds, 0)), weights),
                               dtype=tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_and(tf.equal(real, 0), tf.equal(preds, 1)), weights),
                               dtype=tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_and(tf.equal(real, 1), tf.equal(preds, 0)), weights),
                               dtype=tf.float32))

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-08)
    recall = tp / (tp + fn + 1e-08)
    bacc = (recall + (tn / (tn + fp + 1e-08))) * 0.5
    precision = tp / (tp + fp + 1e-08)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-08)
    mcc = ((tp * tn) - (fp * fn)) / tf.math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-08)

    return bacc, recall, precision, f1, mcc

# Training and Testing/Validation Steps
def run_train_val(FLAGS, epochs, dataset_train, dataset_val, dataset_test, model, optimizer,
                  loss_obj, loss_opt, train_loss_obj, train_acc_obj, train_recall_obj, train_precision_obj, train_f1_obj,
                  train_mcc_obj, val_loss_obj, val_acc_obj, val_recall_obj, val_precision_obj, val_f1_obj,
                  val_mcc_obj, test_loss_obj, test_acc_obj, test_recall_obj, test_precision_obj, test_f1_obj,
                  test_mcc_obj, checkpoint, checkpoint_manager, es_number=50, reset_checkpoint=True):
    
    
    """
    Args:
    - FLAGS: arguments object
    - epochs[int]: number of epochs
    - dataset_train [TF Dataset]: [train_protein_data, train_smiles_data, train_target, train_weights]
    - dataset_val [TF Dataset]: [val_protein_data, val_smiles_data, val_target, val_weights]
    - dataset_test [TF Dataset]: [test_protein_data, test_smiles_data, test_target, test_weights]
    - model [TF Model]: Model Architecture
    - optimizer [TF Optimizer]: optimizer function
    - loss_opt [string]: loss function option and class weights
    - train_loss_obj [TF Metric]: training loss object to compute loss over all batches at each epoch
    - train_acc_obj [TF Metric]: training accuracy object to compute accuracy over all batches at each epoch
    - train_recall_obj [TF Metric]: training recall object to compute loss over all batches at each epoch
    - train_precision_obj [TF Metric]: training precision object to compute accuracy over all batches at each epoch
    - train_f1_obj [TF Metric]: training f1-score object to compute accuracy over all batches at each epoch
    - train_mcc_obj [TF Metric]: training mcc object to compute accuracy over all batches at each epoch
    - val_loss_obj [TF Metric]: validation loss object to compute loss over all batches at each epoch
    - val_acc_obj [TF Metric]: validation accuracy object to compute accuracy over all batches at each epoch
    - val_recall_obj [TF Metric]: validation recall object to compute loss over all batches at each epoch
    - val_precision_obj [TF Metric]: validation precision object to compute accuracy over all batches at each epoch
    - val_f1_obj [TF Metric]: validation f1-score object to compute accuracy over all batches at each epoch
    - val_mcc_obj [TF Metric]: validation mcc object to compute accuracy over all batches at each epoch
    - test_loss_obj [TF Metric]: testing loss object to compute loss over all batches at each epoch
    - test_acc_obj [TF Metric]: testing accuracy object to compute accuracy over all batches at each epoch
    - test_recall_obj [TF Metric]: testing recall object to compute loss over all batches at each epoch
    - test_precision_obj [TF Metric]: testing precision object to compute accuracy over all batches at each epoch
    - test_f1_obj [TF Metric]: testing f1-score object to compute accuracy over all batches at each epoch
    - test_mcc_obj [TF Metric]: testing mcc object to compute accuracy over all batches at each epoch
    - checkpoint [TF Train Checkpoint]: checkpoint object
    - checkpoint_manager [ TF Train Checkpoint Manager]: checkpoint object manager
    - es_number [int]: early stopping patience parameter
    - reset_checkpoint [bool]: Train from scratch or start from a saved checkpoint

    """

    if not reset_checkpoint:
        if checkpoint_manager.latest_checkpoint:
            print("Restores from {}".format(checkpoint_manager.latest_checkpoint))
            checkpoint.restore(checkpoint_manager.latest_checkpoint)

        else:
            print("Start from scratch")

    else:
        print("Reset and start from scratch")

    @tf.function
    def train_step(x1, x2, y, weights, model, optimizer, loss_obj, loss_opt, train_loss_obj, train_acc_obj,
                   train_recall_obj, train_precision_obj, train_f1_obj, train_mcc_obj):

        model_variables = model.trainable_variables

        with tf.GradientTape() as tape:
            logits = model((x1, x2), training=True)
            loss_value = loss_function(loss_obj, y, logits, weights, loss_opt)

        grads = tape.gradient(loss_value, model_variables)

        optimizer.apply_gradients(zip(grads, model_variables))

        train_loss_obj(loss_value)
        acc, recall, precision, f1, mcc = metrics_function(y, logits, weights)
        train_acc_obj(acc)
        train_recall_obj(recall)
        train_precision_obj(precision)
        train_f1_obj(f1)
        train_mcc_obj(mcc)

    @tf.function
    def val_step(x1, x2, y, weights, model, loss_obj, loss_opt, val_loss_obj, val_acc_obj, val_recall_obj, val_precision_obj,
                  val_f1_obj, val_mcc_obj):

        logits = model((x1, x2), training=False)
        loss_value = loss_function(loss_obj, y, logits, weights, loss_opt)
        val_loss_obj(loss_value)
        acc, recall, precision, f1, mcc = metrics_function(y, logits, weights)
        val_acc_obj(acc)
        val_recall_obj(recall)
        val_precision_obj(precision)
        val_f1_obj(f1)
        val_mcc_obj(mcc)

    @tf.function
    def test_step(x1, x2, y, weights, model, loss_obj, loss_opt, test_loss_obj, test_acc_obj, test_recall_obj, test_precision_obj,
                  test_f1_obj, test_mcc_obj):

        logits = model((x1, x2), training=False)
        loss_value = loss_function(loss_obj, y, logits, weights, loss_opt)
        test_loss_obj(loss_value)
        acc, recall, precision, f1, mcc = metrics_function(y, logits, weights)
        test_acc_obj(acc)
        test_recall_obj(recall)
        test_precision_obj(precision)
        test_f1_obj(f1)
        test_mcc_obj(mcc)

    best_epoch = 0
    best_metric = -2.0
    es_count = 0

    for epoch in range(epochs):
        print('Epoch: %d' % epoch)
        start_time = time.time()

        # Training States
        train_loss_obj.reset_states()
        train_acc_obj.reset_states()
        train_recall_obj.reset_states()
        train_precision_obj.reset_states()
        train_f1_obj.reset_states()
        train_mcc_obj.reset_states()

        # Valiation States
        val_loss_obj.reset_states()
        val_acc_obj.reset_states()
        val_recall_obj.reset_states()
        val_precision_obj.reset_states()
        val_f1_obj.reset_states()
        val_mcc_obj.reset_states()

        # Testing States
        test_loss_obj.reset_states()
        test_acc_obj.reset_states()
        test_recall_obj.reset_states()
        test_precision_obj.reset_states()
        test_f1_obj.reset_states()
        test_mcc_obj.reset_states()

        for step, (prot_input, smiles_input, target, weights) in enumerate(dataset_train):
            train_step(prot_input, smiles_input, target, weights, model, optimizer,
                       loss_obj, loss_opt, train_loss_obj,
                       train_acc_obj,
                       train_recall_obj, train_precision_obj, train_f1_obj, train_mcc_obj)

            if step % 1000 == 0:
                print(("Epoch: %d, Step: %d, Train Loss: %0.4f, Train Acc: %0.4f, Train Recall: %0.4f, "
                       "Train Precision: %0.4f, " +
                       "Train F1: %0.4f, Train MCC: %0.4f") % (
                          epoch, step, float(train_loss_obj.result()), float(train_acc_obj.result()),
                          float(train_recall_obj.result()), float(train_precision_obj.result()),
                          float(train_f1_obj.result()), float(train_mcc_obj.result())))

        for step, (prot_input, smiles_input, target, weights) in enumerate(dataset_val):
            test_step(prot_input, smiles_input, target, weights, model, loss_obj, loss_opt, val_loss_obj, val_acc_obj,
                      val_recall_obj, val_precision_obj, val_f1_obj, val_mcc_obj)

        end_time = time.time() - start_time

        for step, (prot_input, smiles_input, target, weights) in enumerate(dataset_test):
            test_step(prot_input, smiles_input, target, weights, model, loss_obj, loss_opt, test_loss_obj, test_acc_obj,
                      test_recall_obj, test_precision_obj, test_f1_obj, test_mcc_obj)

        print("----------//----------")
        print(("Time Taken: %0.2f, Train Loss: %0.4f, Train Acc: %0.4f, Train Recall: %0.4f, Train Precision: %0.4f, " +
               "Train F1: %0.4f, Train MCC: %0.4f, Val Loss: %0.4f, Val Acc: %0.4f, Val Recall: %0.4f, Val Precision: %0.4f, " +
               "Val F1: %0.4f, Val MCC: %0.4f, Test Loss: %0.4f, Test Acc: %0.4f, Test Recall: %0.4f, Test Precision: %0.4f, " +
               "Test F1: %0.4f, Test MCC: %0.4f") % (
                  end_time, float(train_loss_obj.result()), float(train_acc_obj.result()),
                  float(train_recall_obj.result()), float(train_precision_obj.result()),
                  float(train_f1_obj.result()), float(train_mcc_obj.result()), float(val_loss_obj.result()),
                  float(val_acc_obj.result()), float(val_recall_obj.result()), float(val_precision_obj.result()),
                  float(val_f1_obj.result()), float(val_mcc_obj.result()), float(test_loss_obj.result()),
                  float(test_acc_obj.result()), float(test_recall_obj.result()), float(test_precision_obj.result()),
                  float(test_f1_obj.result()), float(test_mcc_obj.result())))

        checkpoint.step.assign_add(1)
        es_count += 1

        if float(val_mcc_obj.result()) > best_metric and float(val_mcc_obj.result()) - best_metric >= 0.001:
            es_count = 0
            best_metric = float(val_mcc_obj.result())
            best_epoch = epoch
            checkpoint_manager.save(checkpoint_number=int(best_epoch))
            print("----------//----------")
            print('Validation MCC Improved at Epoch: %d' % epoch)

            logging(FLAGS,
                    ("Epoch: %d, Train Loss: %0.4f, Train Acc: %0.4f, Train Recall: %0.4f, Train Precision: %0.4f, " +
                     "Train F1: %0.4f, Train MCC: %0.4f, Val Loss: %0.4f, Val Acc: %0.4f, Val Recall: %0.4f, Val Precision: %0.4f, " +
                     "Val F1: %0.4f, Val MCC: %0.4f, Test Loss: %0.4f, Test Acc: %0.4f, Test Recall: %0.4f, Test Precision: %0.4f, " +
               "Test F1: %0.4f, Test MCC: %0.4f") % (
                        best_epoch, float(train_loss_obj.result()), float(train_acc_obj.result()),
                        float(train_recall_obj.result()), float(train_precision_obj.result()),
                        float(train_f1_obj.result()), float(train_mcc_obj.result()), float(val_loss_obj.result()),
                        float(val_acc_obj.result()), float(val_recall_obj.result()), float(val_precision_obj.result()),
                        float(val_f1_obj.result()), float(val_mcc_obj.result()), float(test_loss_obj.result()),
                  float(test_acc_obj.result()), float(test_recall_obj.result()), float(test_precision_obj.result()),
                  float(test_f1_obj.result()), float(test_mcc_obj.result())))
        else:
            print('No Improvement on validation mcc since epoch: %d' % best_epoch)

        if es_count == es_number:
            break



