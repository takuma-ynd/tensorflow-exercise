#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
import logreg_online as util
import ipdb

# mini-batchを使った学習
# feed_dictする代わりに、全データをTensorFlowに渡しておいて、
# 実行時にbatchを作ってもらうのがポイント

def pad(bumpy_lists):
    """Add zero-padding to bumpy lists
    Arg:
    bumpy_lists:
    - rank2, bumpy list

    Return:
    - rank2, lists of same length with zero-padding
    
    Ex:
    input: [[1,2,3,4,5], [1,2], [1,2,3]]
    output: [[1,2,3,4,5], [1,2,0,0,0], [1,2,3,0,0]]
    """
    def pad_list(rank1_list):
        return rank1_list + [0 for _ in range(maxlen - len(rank1_list))]

    maxlen = max(len(list_) for list_ in bumpy_lists)
    return [pad_list(list_) for list_ in bumpy_lists]

    
def generate_batches(labels, fvs, batch_size=10, shuffle=False):
    """generate batches from fvs and labels
    Args:
    input_fvs: Tensor
    input_labels: Tensor

    Returns:
    tuple of Tensors (labels and fvs)
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
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
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

        except tf.errors.OutOfRangeError:
            # batch_sizeを10にしてもこの例外送出されるのはマジで謎
            print("OutOfRangeError but no problem !!")

        finally:
            coord.request_stop()

        coord.join(threads)
    return

if __name__ == "__main__":
    dim = 50
    batch_size = 10 # バッチ処理
    NUM_EPOCS = 30
    SHUFFLE = True

    assert len(sys.argv) > 2, "arg1: train_file, arg2: test_file"
    train_path = sys.argv[1]
    test_path = sys.argv[2]

    # ファイルをオープン
    with open(train_path) as f:
        train_text = f.read().strip()

    with open(test_path) as f:
        test_text = f.read().strip()

    train_data, vocab_size = util.read_data(train_text,-1)
    test_data, _ = util.read_data(test_text, vocab_size)

    # グラフ作成
    mixed_graph = tf.Graph()
    with mixed_graph.as_default():

        # バッチ生成とグラフ構築
        input_fvs = tf.placeholder(tf.int32, shape=[None, None], name="input_fvs") # [batch_size x dim]
        input_labels = tf.placeholder(tf.int32, shape=None, name="input_labels") # [batch_size]
        fvs = input_fvs
        signed_labels = input_labels
        labels = tf.div((signed_labels + 1), 2)  # {-1,1} --> {0,1}

        # 変数の定義
        weight = tf.Variable(tf.random_uniform([dim, 2]), name="weight")
        bias = tf.Variable(tf.random_uniform([1, 2]), name="bias") # valid size: [batch_size x 2]
        embeddings = tf.Variable(tf.random_uniform([vocab_size, dim]), name="embeddings")

        # 必要な変数
        vectors = tf.nn.embedding_lookup(embeddings, fvs)
        ave_vectors = tf.reduce_mean(vectors, axis=1)

        # logistic regression の計算
        logits = tf.add(tf.matmul(ave_vectors, weight), bias)
        y = tf.nn.softmax(logits)

        # tf.one_hot(indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None)
        one_hot = tf.one_hot(labels, 2)
        cross_entropy = -tf.reduce_sum(tf.multiply(one_hot, tf.log(y)))

        # トレーニングの設定
        optimizer = tf.train.AdamOptimizer() # AdamOptimizerをoptimizerとして設定
        train_op = optimizer.minimize(cross_entropy) # train operationを定義

        # 評価グラフ
        predicted_labels = tf.argmax(y, axis=1)
        accuracy, accuracy_update_op = tf.metrics.accuracy(labels, predicted_labels)
        precision, precision_update_op = tf.metrics.precision(labels, predicted_labels)
        recall, recall_update_op = tf.metrics.recall(labels, predicted_labels)

    with tf.Session(graph=mixed_graph) as sess:

        ### Training ###

        # tfに食わせるデータの取得
        labels, fvs = train_data
        fvs = pad(fvs) # zero-padding.

        # batchの作成
        train_batches = list(generate_batches(labels, fvs, batch_size=10, shuffle=SHUFFLE))

        # 初期化
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        # 各バッチ毎にトレーニング
        for epoch in range(NUM_EPOCS):
            for i, (batch_labels, batch_fvs) in enumerate(train_batches):
                feed = {input_labels:batch_labels, input_fvs:batch_fvs}
                _, loss = sess.run([train_op, cross_entropy], feed_dict=feed)
                print("epoch:{}\ttrain_data:{}\tcross_entropy:{}".format(epoch, i, loss))
        print("--- training finished ---")

        ### Evaluation ###

        # tfに食わせるデータの取得
        labels, fvs = test_data
        fvs = pad(fvs) # zero-padding.

        # batchの作成
        test_batches = list(generate_batches(labels, fvs, batch_size=10))

        # 初期化
        eval_init_op = tf.local_variables_initializer()
        sess.run(eval_init_op)

        # 各バッチ毎に評価
        for i, (batch_labels, batch_fvs) in enumerate(test_batches):
            feed = {input_labels:batch_labels, input_fvs:batch_fvs}
            acc, pre, rec = sess.run([
                accuracy_update_op,
                precision_update_op,
                recall_update_op],feed_dict=feed)
            print("iter:{}\tacc:{}\tpre:{}\trec:{}".format(i, acc, pre, rec))
        print("accuracy:", acc)
        print("f-measure:", 2*(pre*rec)/(pre+rec))

