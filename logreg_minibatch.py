#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import tensorflow as tf
import logreg_online as util

# import ipdb

# mini-batchを使った学習
# tf.train.batchを用いて、事前にmini-batchを作成するグラフを作るようにしている
# mini-batch生成のためだけにグラフを作ってsess.runするという、
# かなり変な形になっているので普通にpythonでmini-batch作成したほうが速いと思う。

# memo:
# これこそがwith tf.Graph().as_default(): でグラフを分けることが必要な良い例?
# trainingのセッション中に、現在のグラフに対して初期化等行わずに
# generate_batches(内部で別グラフ作成、別Sessionをsess.run)などを実行可能。
def pad(bumpy_lists):
    """Add zero-padding to bumpy lists
    Arg:
    bumpy_lists:
    - rank2, bumpy list

    Return:
    - lists of same length with zero-padding

    Ex:
    input: [[1,2,3,4,5], [1,2], [1,2,3]]
    output: [[1,2,3,4,5], [1,2,0,0,0], [1,2,3,0,0]]
    """
    def pad_rank1_list(rank1_list):
        """Add zero-padding to a single (rank1) list
        """
        return rank1_list + [0 for _ in range(maxlen - len(rank1_list))]

    maxlen = max(len(list_) for list_ in bumpy_lists)
    return [pad_rank1_list(list_) for list_ in bumpy_lists]

def generate_batches(labels, fvs, batch_size=10, shuffle=False):
    """generate batches from fvs and labels
    Args:
    labels: list of labels
    fvs: list of list(fvs)

    Returns:
    generator: generate mini-batch of (labels, fvs) each time.

    Note:
    An example using tf.train.batches:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_preloaded.py
    """
    print("building batch generation graph...")
    batch_generation_graph = tf.Graph()
    with batch_generation_graph.as_default():
        input_labels, input_fvs = tf.constant(labels), tf.constant(fvs)

        # Queueを作り、batch_size回enqueueしてbatchを作る
        input_label, input_fv = tf.train.slice_input_producer([input_labels, input_fvs], num_epochs=1)

        if shuffle:
            # capacity, min_after_dequeueはてきとう。どれくらいの値にするべきなのかは知らない
            batch_labels, batch_fvs = tf.train.shuffle_batch(
                [input_label, input_fv],
                batch_size,
                capacity=50000,
                min_after_dequeue=10000)
        else:
            batch_labels, batch_fvs = tf.train.batch([input_label, input_fv], batch_size)    

    with tf.Session(graph=batch_generation_graph) as sess:

        # 初期化
        init_op = tf.local_variables_initializer()
        sess.run(init_op)

        # train.batchに必要
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 順にenqueueしてデータを順に学習
        try:
            counter = 0
            while not coord.should_stop():
                counter += 1
                yield sess.run([batch_labels, batch_fvs])

        except tf.errors.OutOfRangeError: # 全データ数がbatch_sizeで割り切れない時に起こるはず
            # batch_sizeを10(割り切れる数)にしてもこの例外送出されるのは謎
            pass

        finally:
            coord.request_stop()

        coord.join(threads)

def build_graph(dim, l2_coef):
    """build forward + evaluation graph
    builds graph and return the object which contains tensors to be used.
    
    Note:
    input_fvs will be zero-padded tensor.
    This graph computes actual sequence length(the number of non-zero values) on the fly. 
    http://danijar.com/variable-sequence-lengths-in-tensorflow/
    """

    mixed_graph = tf.Graph()
    with mixed_graph.as_default():

        # placeholders
        input_fvs = tf.placeholder(tf.int32, shape=[None, None], name="input_fvs") # [batch_size x dim]
        input_labels = tf.placeholder(tf.int32, shape=None, name="input_labels") # [batch_size]
        keep_prob = tf.placeholder(tf.float32) # scalar

        # input_fvs, input_labels --> fvs, labels
        fvs = input_fvs
        signed_labels = input_labels
        labels = tf.div((signed_labels + 1), 2)  # {-1,1} --> {0,1}

        # 変数の定義
        # tf.pad(tensor, [[1次元目のbefore, 1次元目のafter], [2次元目のbefore, 2次元目のafter]])
        weight = tf.Variable(tf.random_uniform([dim, 2]), name="weight")
        bias = tf.Variable(tf.random_uniform([1, 2]), name="bias") # valid size: [batch_size x 2]
        core_embeddings = tf.Variable(tf.random_uniform([vocab_size, dim]), name="core-embeddings")
        embeddings = tf.pad(core_embeddings, [[1, 0], [0, 0]]) # add an additional row for zero-padding

        vectors = tf.nn.embedding_lookup(embeddings, fvs)

        # vectors --> ave_vectors
        # zero-paddingによって単純にreduce_meanできない問題を解決するtrick(http://danijar.com/variable-sequence-lengths-in-tensorflow/)
        used = tf.sign(tf.reduce_max(tf.abs(vectors), axis=2)) # zero_paddingには0, その他には1が載るmaskを作成
        length = tf.reduce_sum(used, axis=1, keep_dims=True) # keep_dimsをつけないとshape:(batch_size,)となり、割り算が不可能に.
        sum_vectors = tf.reduce_sum(vectors, axis=1)
        ave_vectors = sum_vectors / length 

        # logistic regression の計算
        # evaluation時にはkeep_probを1に戻してあげる
        ave_vectors = tf.nn.dropout(ave_vectors, keep_prob)
        logits = tf.add(tf.matmul(ave_vectors, weight), bias)
        y = tf.nn.softmax(logits)

        # tf.one_hot(indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None)
        one_hot = tf.one_hot(labels, depth=2)
        cross_entropy = -tf.reduce_sum(tf.multiply(one_hot, tf.log(y))) + l2_coef * tf.nn.l2_loss(weight)

        # トレーニングの設定
        optimizer = tf.train.AdamOptimizer() # AdamOptimizerをoptimizerとして設定
        train_op = optimizer.minimize(cross_entropy) # train operationを定義

        # 評価グラフ
        predicted_labels = tf.argmax(y, axis=1)
        _, accuracy_update_op = tf.metrics.accuracy(labels, predicted_labels)
        _, precision_update_op = tf.metrics.precision(labels, predicted_labels)
        _, recall_update_op = tf.metrics.recall(labels, predicted_labels)

        # tensorboard用のsummary
        loss_summary = tf.summary.scalar("cross_entropy",cross_entropy)
        merged = tf.summary.merge_all()

    class TrainGraph(object):
        def __init__(self):
            # place_holders
            self.input_fvs = input_fvs
            self.input_labels = input_labels
            self.keep_prob = keep_prob

            # operations
            self.train_op = train_op
            self.accuracy = accuracy_update_op
            self.precision = precision_update_op
            self.recall = recall_update_op

            # additional variable
            self.cross_entropy = cross_entropy

            # output
            self.graph = mixed_graph
            self.predicted_labels = predicted_labels
            self.merged = merged

    return TrainGraph()


if __name__ == "__main__":
    assert len(sys.argv) > 2, "arg1: train_file, arg2: test_file"
    train_path = sys.argv[1]
    test_path = sys.argv[2]

    # tf.flagsの設定。実行時に引数渡すだけで変数の値を買えられるようになる.
    tf.flags.DEFINE_integer("dim", 50, "dimension of embeddings. (default: 50)")
    tf.flags.DEFINE_integer("batch-size", 16, "batch size. (default: 16)")
    tf.flags.DEFINE_float("train-dropout", 0.5, "keep probability of dropout for a training. (default: 0.5)")
    tf.flags.DEFINE_integer("num-epochs", 50, "number of epochs to train. (default: 50)")
    tf.flags.DEFINE_boolean("shuffle", True, "whether or not to shuffle train data. (default: True)")
    tf.flags.DEFINE_float("l2-coef", 1e-08, "coefficient for l2 regurarization.(default: 1e-08)")
    tf.flags.DEFINE_string("logdir", "/tmp/minibatch_train", "log directory for TensorBoard. (default:/tmp/minibatch_train)")
    tf.flags.DEFINE_boolean("eval-log", False, "whether or not to save evaluation data to eval-log-file. (default: Flase)")
    tf.flags.DEFINE_string("eval-log-file", "evaluation-result.log", "path of evaluation log file (default: evaluation-result.log)")
    FLAGS = tf.flags.FLAGS

    # ファイルをオープン
    with open(train_path) as f:
        train_text = f.read().strip()

    with open(test_path) as f:
        test_text = f.read().strip()

    train_data, vocab_size = util.read_data(train_text,-1)
    test_data, _ = util.read_data(test_text, vocab_size)

    graph = build_graph(FLAGS.dim, FLAGS.l2_coef)

    with tf.Session(graph=graph.graph) as sess:
        #example: Fri_Jun__2_16:07:20_2017
        board_name = time.ctime(time.time()).replace(" ", "_")
        tb_logdir = FLAGS.logdir + "/"  + board_name
        print("training log will be summarized in:{}".format(tb_logdir))

        # for tensorboard
        train_writer = tf.summary.FileWriter(tb_logdir, graph=sess.graph)

        ### Training ###

        # tfに食わせるデータの取得
        labels, fvs = train_data
        fvs = pad(fvs) # zero-padding.

        # batchの作成
        train_batches = list(generate_batches(labels, fvs, batch_size=FLAGS.batch_size, shuffle=FLAGS.shuffle))
        num_batches = len(train_batches)

        # 初期化
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        # 各バッチ毎にトレーニング
        for epoch in range(FLAGS.num_epochs):
            for i, (batch_labels, batch_fvs) in enumerate(train_batches):
                feed = {graph.input_labels:batch_labels,
                        graph.input_fvs:batch_fvs,
                        graph.keep_prob:FLAGS.train_dropout}
                _, loss, summary = sess.run([
                    graph.train_op,
                    graph.cross_entropy,
                    graph.merged], feed_dict=feed)

                # batchの1/10を学習する毎にログを取る
                if i % (num_batches // 10)== 0:
                    print("epoch:{}\ttrain_data:{}\tcross_entropy:{}".format(epoch, i, loss))
                    train_writer.add_summary(summary, global_step=(epoch*num_batches + i))

        print("--- training finished ---")

        ### Evaluation ###

        # tfに食わせるデータの取得
        labels, fvs = test_data
        fvs = pad(fvs) # zero-padding.

        # batchの作成
        test_batches = generate_batches(labels, fvs, batch_size=10)

        # 初期化
        eval_init_op = tf.local_variables_initializer()
        sess.run(eval_init_op)
        eval_dropout = 1.0

        # 各バッチ毎に評価
        for i, (batch_labels, batch_fvs) in enumerate(test_batches):
            feed = {graph.input_labels:batch_labels,
                    graph.input_fvs:batch_fvs,
                    graph.keep_prob:eval_dropout}
            acc, pre, rec = sess.run([
                graph.accuracy,
                graph.precision,
                graph.recall],feed_dict=feed)
        print("acc:{}\tpre:{}\trec:{}".format(acc, pre, rec))
        f_measure = 2*(pre*rec)/(pre+rec)
        print("f-measure:", f_measure)

        if FLAGS.eval_log:
            with open(FLAGS.eval_logfile, "a") as f:
                f.write("num-epochs:{}\tl2_coef:{}\ttrain_dropout:{}\tbatch-size:{}\tdim:{}\tacc:{:.4}\tf:{:.4}\n".format(FLAGS.dim, FLAGS.l2_coef, FLAGS.train_dropout, FLAGS.batch_size, FLAGS.dim, acc, f_measure))
