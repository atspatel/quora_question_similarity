""" Let's find similarity between 2 questions on Quora.

"""

__all__ = ['QuestionSimilarity', 'InputData', 'Sent2Vector']
__version__ = '0.1'
__author__ = 'Atish Patel'


import os
import dill

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import time
import datetime

class InputData():
    # to store all tf input

    def __init__(self, max_sent_token_length,
                    l2_reg_lambda = 0.0):
        self.input_x1 = tf.placeholder(tf.int32, [None, max_sent_token_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, max_sent_token_length], name="input_x2")

        self.input_seq_len_x1 = tf.placeholder(tf.int32, [None], name="input_seq_len_x1")
        self.input_seq_len_x2 = tf.placeholder(tf.int32, [None], name="input_seq_len_x2")

        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")

        self.batch_size = tf.shape(self.input_x1)[0]
        # self.len_input_x2 = tf.shape(self.input_x2)[0]

        self.dropout_keep_prob = tf.placeholder(tf.float32, name= 'dropout_keep_prob')
        self.lstm_output_keep_prob = tf.placeholder(tf.float32, name= 'lstm_output_keep_prob')

        self.l2_reg_lambda = l2_reg_lambda
        self.l2_loss = 0

class Sent2Vector():

    """
    Let's try to learn vector representation of sentence.
    """

    def __init__(self, InputData_Object,
                word_embedding_size, word_vocab_len,
                word_rnn_size, word_num_layers,
                hidden_layer_neurons, output_dim):

        concated_sent = tf.concat([InputData_Object.input_x1, InputData_Object.input_x2], axis = 0, name="concated_sent")
        concated_seq_len = tf.concat([InputData_Object.input_seq_len_x1, InputData_Object.input_seq_len_x2], axis = 0, name="concated_seq_len")
        with tf.name_scope("word_embedding"):
            self.word_embedding_mat = tf.Variable(
                tf.random_uniform([word_vocab_len, word_embedding_size], -1.0, 1.0),
                name="word_embedding_mat")
            self.embedded_word = tf.nn.embedding_lookup(self.word_embedding_mat, concated_sent)
            self.embedded_word_dropped = tf.nn.dropout(self.embedded_word, InputData_Object.dropout_keep_prob, name="embedded_word_dropped")
            print ("shape of embedded_word : ",self.embedded_word_dropped.get_shape())

        def extract_axis_1(data, ind):
            batch_range = tf.range(tf.shape(data)[0])
            ind = tf.subtract(ind,1)
            new_ind = tf.where(ind > 0, ind, tf.zeros_like(ind))
            indices = tf.stack([batch_range, new_ind], axis=1)
            res = tf.gather_nd(data, indices)
            return res

        def create_lstm_cell(rnn_size_input):
            lstm_cell = rnn.BasicLSTMCell(rnn_size_input, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
            lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob = InputData_Object.lstm_output_keep_prob)
            return lstm_cell

        def temporal_average(input_matrix, sequence_length):
            lengths_transposed = tf.expand_dims(sequence_length, 1)
            range_row = tf.expand_dims(tf.range(0, tf.shape(input_matrix)[1], 1), 0)
            mask = tf.expand_dims(tf.cast(tf.less(range_row, lengths_transposed), tf.float32), 2)
            sequence_length = tf.expand_dims(tf.cast(sequence_length, tf.float32), 1)
            return tf.div((tf.reduce_sum(tf.multiply(input_matrix, mask), 1)), sequence_length)

        with tf.variable_scope("forward_LSTM"):
            fw_cell = rnn.MultiRNNCell([create_lstm_cell(word_rnn_size) for _ in range(word_num_layers)], state_is_tuple=True)
            output_fw, _ = tf.nn.dynamic_rnn(fw_cell,self.embedded_word_dropped,dtype=tf.float32, sequence_length= concated_seq_len)
            self.word_lstm_output = extract_axis_1(output_fw, concated_seq_len)
            print ("shape of word_lstm_output : ",self.word_lstm_output.get_shape())

        InputData_Object.l2_loss = 0.0
        hidden_layer_input = self.word_lstm_output
        with tf.name_scope("dense_network"):
            for i, hidden_nurons in enumerate(hidden_layer_neurons):
                with tf.variable_scope("hidden_layer_%s" % i):
                    input_dim = hidden_layer_input.get_shape().as_list()[1]
                    
                    Weight_mat = tf.get_variable(
                        "weight_mat_%s" % i,
                        shape = [input_dim, hidden_nurons],
                        initializer = tf.contrib.layers.xavier_initializer())

                    InputData_Object.l2_loss += tf.nn.l2_loss(Weight_mat)

                    bais_mat = tf.Variable(tf.constant(0.1, shape=[hidden_nurons]), name = "bais_mat_%s" % i)
                    hidden_layer_input = tf.nn.relu(tf.nn.xw_plus_b(hidden_layer_input, Weight_mat, bais_mat))
                    hidden_layer_input = tf.nn.dropout(hidden_layer_input, InputData_Object.dropout_keep_prob, name = "hidden_out_%s" % i)
                    print ("hidden_out shape: ",hidden_layer_input.get_shape())
        
        self.hidden_out = hidden_layer_input

        ## output_layer
        with tf.name_scope('output'):
            input_dim = hidden_layer_input.get_shape().as_list()[1]
            Weight_mat_out = tf.get_variable("output_w",
                shape = [input_dim, output_dim],
                initializer = tf.contrib.layers.xavier_initializer())

            InputData_Object.l2_loss += tf.nn.l2_loss(Weight_mat)

            bais_mat_out = tf.Variable(tf.constant(0.1, shape=[output_dim]), name = "output_b")
            sent_vectors = tf.nn.xw_plus_b(self.hidden_out, Weight_mat_out, bais_mat_out, name = "output_score")
            normalized_sent_vectors = tf.nn.l2_normalize(sent_vectors, 1, name="normalized_sent_vectors")
        self.sent_vec1, self.sent_vec2 = tf.split(normalized_sent_vectors, [InputData_Object.batch_size, InputData_Object.batch_size], 0)

class QuestionSimilarity():

    """
    find similarity between two sentences.
    Uses a LSTM network followed by DNN.

    Calculating the vector using Sent2Vector module and cosine taking similarity between 2 vectors
    """
    def __init__(self, InputData_Object,
                Sent2Vector_Object, threshold):
        vector_sent1 = Sent2Vector_Object.sent_vec1
        vector_sent2 = Sent2Vector_Object.sent_vec2

        self.cosine_scores = tf.reduce_sum(tf.multiply(vector_sent1, vector_sent2), axis = 1)
        print ("cosine_scores shape: ",self.cosine_scores.get_shape())

        with tf.name_scope('predictions'):
            self.predictions = tf.where(self.cosine_scores < threshold, tf.zeros_like(self.cosine_scores), tf.ones_like(self.cosine_scores), name = 'predictions')

        with tf.name_scope('loss'):
            squared_cosine = tf.square(self.cosine_scores)
            y = tf.cast(InputData_Object.input_y, tf.float32)
            error_loss = tf.reduce_mean((y * (1 - squared_cosine))/2.0 + (1-y)*self.predictions*squared_cosine, name = 'error_loss')
            l2_loss_batch = InputData_Object.l2_reg_lambda*InputData_Object.l2_loss

            self.loss = error_loss + l2_loss_batch

        with tf.name_scope('metrics'):
            predictions_int = tf.cast(self.predictions, tf.int32)
            self.tp = tf.count_nonzero(predictions_int * InputData_Object.input_y)
            self.tn = tf.count_nonzero((predictions_int - 1) * (InputData_Object.input_y - 1))
            self.fp = tf.count_nonzero(predictions_int * (InputData_Object.input_y - 1))
            self.fn = tf.count_nonzero((predictions_int - 1) * InputData_Object.input_y)

            one_float = tf.constant(0.001, dtype = tf.float64)
            one_int = tf.constant(1, dtype = tf.int64)
            self.accuracy = tf.divide((self.tp + self.tn), (self.tp + self.fp + self.fn + self.tn + one_int))
            self.precision = tf.divide(self.tp , (self.tp + self.fp + one_int))
            self.recall = tf.divide(self.tp , (self.tp + self.fn + one_int))
            self.fscore = tf.divide(( tf.constant(2, dtype = tf.float64) * self.precision * self.recall) , (self.precision + self.recall + one_float))

if __name__=="__main__":
    import data_helper
    import dill

    TRAIN_FILE = "../data/train.csv"

    tf.flags.DEFINE_string("model_name","question_similarity","Name of model to save")

    tf.flags.DEFINE_integer("word_embedding_size", 100, "Dimensionality of word embedding (default: 128)")
    tf.flags.DEFINE_integer("word_rnn_size", 128, "size of rnn layers in word lstm (default: 128)")
    tf.flags.DEFINE_integer("word_num_layers", 2, "Number of layers in word lstm")

    tf.flags.DEFINE_string("hidden_layer_neurons", '128', "Dense Network Structure")
    tf.flags.DEFINE_integer("output_dim", 64, "output_dim")

    tf.flags.DEFINE_float("dropout_keep_prob", 0.9, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("lstm_output_keep_prob", 0.8, "LSTM Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 0.00001, "L2 regularizaion lambda (default: 0.0)")

    tf.flags.DEFINE_float("threshold", 0.7, "Dropout keep probability (default: 0.5)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 1000, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    Flags_Dict = {}
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
        Flags_Dict[attr] = value
    print("")


    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.model_name))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    Flags_Dict['out_dir']=out_dir
    Flags_Dict['model_path']=out_dir

    

    print ("Preprocessing Data.......")
    data = data_helper.ReadCsv(TRAIN_FILE, True)
    train_set, dev_set, id_sentence_mapping, all_training_tokens, max_sent_token_len = data_helper.pre_processing_info(data, 0.1)
    CreateVector_Object = data_helper.CreateVector(all_training_tokens, max_sent_token_len, id_sentence_mapping)

    csv_title = ['id_1', 'id_2', 'label']
    train_file = '../data/train_data.csv'
    data_helper.WriteCsv(train_set, train_file, csv_title)

    dev_file = '../data/dev_data.csv'
    data_helper.WriteCsv(dev_set, dev_file, csv_title)


    dill.dump(Flags_Dict,open(os.path.join(out_dir, "config.pkl"),'wb')) 
    dill.dump(CreateVector_Object,open(os.path.join(out_dir, "vector_class.pkl"),'wb')) 

    best_f1 = 0
    precision_at_best_f1 = 0
    recall_at_best_f1 = 0


    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            InputData_Object = InputData(
                        max_sent_token_length = max_sent_token_len,
                        l2_reg_lambda = 0.0)
            Sent2Vector_Object = Sent2Vector(
                        InputData_Object = InputData_Object,
                        word_embedding_size = FLAGS.word_embedding_size,
                        word_vocab_len = max(CreateVector_Object.WordVocabProcessor_Object.index_dict.values()) + 1,
                        word_rnn_size = FLAGS.word_rnn_size,
                        word_num_layers = FLAGS.word_num_layers,
                        hidden_layer_neurons = list(map(int, FLAGS.hidden_layer_neurons.split(","))),
                        output_dim = FLAGS.output_dim)

            QuestionSimilarity_Object = QuestionSimilarity(InputData_Object, Sent2Vector_Object, FLAGS.threshold)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(0.01)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(QuestionSimilarity_Object.loss, tvars), 10)
            grads_and_vars = zip(grads,tvars)
            train_op = optimizer.apply_gradients(grads_and_vars,global_step=global_step)

            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)

            grad_summaries_merged = tf.summary.merge(grad_summaries) 
            timestamp = str(int(time.time()))
            print("Writing to {}\n".format(out_dir))

            loss_summary = tf.summary.scalar("loss", QuestionSimilarity_Object.loss)
            train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)


            dev_summary_op = tf.summary.merge([loss_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            saver = tf.train.Saver(tf.global_variables())
            sess.run(tf.global_variables_initializer())

            glove_embedding_mat = data_helper.get_glove_mat(all_training_tokens, CreateVector_Object.WordVocabProcessor_Object)
            sess.run(Sent2Vector_Object.word_embedding_mat.assign(glove_embedding_mat))

            del glove_embedding_mat

            def train_step(x_batch, y_batch):
                input_mat_sent1, input_mat_sent2, input_seq_len_sent1, input_seq_len_sent2 = zip(*x_batch)
                feed_dict = {
                        InputData_Object.input_x1 : input_mat_sent1,
                        InputData_Object.input_x2 : input_mat_sent2,
                        InputData_Object.input_seq_len_x1 : input_seq_len_sent2,
                        InputData_Object.input_seq_len_x2 : input_seq_len_sent2,

                        InputData_Object.input_y : y_batch,

                        InputData_Object.dropout_keep_prob: FLAGS.dropout_keep_prob,
                        InputData_Object.lstm_output_keep_prob : FLAGS.lstm_output_keep_prob}
                _, step,summaries,loss, accuracy, precision, recall, fscore = sess.run(
                    [train_op, global_step,train_summary_op, QuestionSimilarity_Object.loss,
                        QuestionSimilarity_Object.accuracy, QuestionSimilarity_Object.precision,
                        QuestionSimilarity_Object.recall, QuestionSimilarity_Object.fscore],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, accuracy {:g},  precision {:g}, recall {:g}, f1_score {:g}".format(time_str, step, loss, accuracy, precision, recall, fscore))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch):
                input_mat_sent1, input_mat_sent2, input_seq_len_sent1, input_seq_len_sent2 = zip(*x_batch)
                feed_dict = {
                        InputData_Object.input_x1 : input_mat_sent1,
                        InputData_Object.input_x2 : input_mat_sent2,
                        InputData_Object.input_seq_len_x1 : input_seq_len_sent2,
                        InputData_Object.input_seq_len_x2 : input_seq_len_sent2,

                        InputData_Object.input_y : y_batch,

                        InputData_Object.dropout_keep_prob: 1.0,
                        InputData_Object.lstm_output_keep_prob : 1.0}
                step,summaries,loss, accuracy, precision, recall, fscore = sess.run(
                    [global_step,train_summary_op, QuestionSimilarity_Object.loss,
                        QuestionSimilarity_Object.accuracy, QuestionSimilarity_Object.precision,
                        QuestionSimilarity_Object.recall, QuestionSimilarity_Object.fscore],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, accuracy {:g},  precision {:g}, recall {:g}, f1_score {:g}".format(time_str, step, loss, accuracy, precision, recall, fscore))
                dev_summary_writer.add_summary(summaries, step)
                return accuracy, precision, recall, fscore, loss

            batches = data_helper.batch_iter(train_set, FLAGS.batch_size, FLAGS.num_epochs, True)
            for i, batch in enumerate(batches):
                x_batch, y_batch = CreateVector_Object.create_train_vec(batch)
                train_step(x_batch, y_batch)

                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    dev_accuracy_list = []
                    dev_precision_list = []
                    dev_recall_list = []
                    dev_fscore_list = []
                    dev_loss_list = []

                    dev_batches = data_helper.batch_iter(dev_set, FLAGS.batch_size, 1, True)
                    for dev_batch in dev_batches:
                        dev_input_x, dev_input_y = CreateVector_Object.create_train_vec(dev_batch)
                        dev_accuracy, dev_precision, dev_recall, dev_fscore, dev_loss = dev_step(dev_input_x, dev_input_y)
                        dev_accuracy_list.append(dev_accuracy)
                        dev_precision_list.append(dev_precision)
                        dev_recall_list.append(dev_recall)
                        dev_fscore_list.append(dev_fscore)
                        dev_loss_list.append(dev_loss)

                    dev_accuracy = np.mean(np.array(dev_accuracy_list))
                    dev_precision = np.mean(np.array(dev_precision_list))
                    dev_recall = np.mean(np.array(dev_recall_list))
                    dev_f1 = np.mean(np.array(dev_fscore_list))
                    dev_loss = np.mean(np.array(dev_loss_list))

                if current_step % FLAGS.checkpoint_every == 0:
                    if dev_f1 > best_f1:
                        best_f1 = dev_f1
                        precision_at_best_f1 = dev_precision
                        recall_at_best_f1 = dev_recall

                    # if dev_f1 > best_f1 - 0.005:
                        print ("dev_f1 : %f , best_f1 : %f , precision_at_best_f1 : %f , recall_at_best_f1 : %f"%(dev_f1, best_f1, precision_at_best_f1, recall_at_best_f1))
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
