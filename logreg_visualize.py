# -*- coding: utf-8 -*-
import sys
import tensorflow as tf
import ipdb
import random
import logreg_online as util

if __name__ == "__main__":
    assert len(sys.argv) > 2
    SHUFFLE = True
    NUM_EPOCS = 5

    train_file_path = sys.argv[1]
    eval_file_path = sys.argv[2]
    dim = 50

    # ファイルをオープン
    with open(train_file_path) as f:
        train_txt = f.read().strip()

    with open(eval_file_path) as f:
        test_txt = f.read().strip()

    # tfに食わせるデータの取得
    train_data, vocab_size = util.read_data(train_txt, -1)
    test_data, _ = util.read_data(test_txt, vocab_size)


    ### グラフの作成 ###
    mixed_graph = tf.Graph()
    with mixed_graph.as_default():

        # 変数の定義
        weight = tf.Variable(tf.random_uniform([dim, 2])) # 2値分類ゆえ,[dim x 2]
        bias = tf.Variable(tf.random_uniform([1, 2])) # 2値分類ゆえ,[1 x 2]
        embeddings = tf.Variable(tf.random_uniform([vocab_size, dim]), name="embeddings")

        # placeholderの定義
        # この記述でindicesはリストになる, shape=Noneに注意。indicesの数はmax_indexに関係ない
        indices = tf.placeholder(tf.int32, shape=None) # tf.shape(indices): [847]
        signed_label = tf.placeholder(tf.int32, shape=None) # tf.shape(signed_label): [] (Scalar)
        label = tf.div((signed_label + 1), 2)  # {-1,1} --> {0,1}

        # 必要な変数
        vectors = tf.nn.embedding_lookup(embeddings, indices) # indicesはリスト. tf.shape(vectors):[847, 50]

        # keep_dims=Trueを立てないと、rankが1になっちまう(「行列」ではなく、「ベクトル」になる感じ)
        ave_vector = tf.reduce_mean(vectors, axis=0, keep_dims=True) # tf.shape(ave_vector):[1, 50]

        # logistic regression の計算
        logit = tf.add(tf.matmul(ave_vector, weight), bias) # tf.shape(logit): [1,2]
        y = tf.nn.softmax(logit) # tf.shape(y): [1,2]

        # tf.one_hot(indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None)
        one_hot = tf.one_hot(label, 2) # tf.shape(one_hot): [2]
        cross_entropy = -tf.reduce_sum(tf.multiply(one_hot, tf.log(y))) # tf.shape(cross_entropy):[] (scalar)

        # トレーニングの設定
        optimizer = tf.train.AdamOptimizer() # AdamOptimizerをoptimizerとして設定
        train_op = optimizer.minimize(cross_entropy) # train operationを定義

        # 評価グラフ
        predicted_label = tf.argmax(y, axis=1)
        accuracy, accuracy_update_op = tf.metrics.accuracy(label, predicted_label)
        precision, precision_update_op = tf.metrics.precision(label, predicted_label)
        recall, recall_update_op = tf.metrics.recall(label, predicted_label)

        # tensorboard用のsummary
        summary01 = tf.summary.scalar("cross_entropy",cross_entropy)
        summary02 = tf.summary.histogram("weight",weight)
        summary03 = tf.summary.scalar("abs_weight",tf.reduce_sum(tf.square(weight)))
        merged = tf.summary.merge_all()

    with tf.Session(graph=mixed_graph) as sess:

        # for tensorboard
        train_writer = tf.summary.FileWriter("/tmp/tensorflow_train", graph=sess.graph)

        ### Training ###

        # 初期化処理
        train_init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        sess.run(train_init_op)

        print("--- training ---")

        # train dataをシャッフルする
        if SHUFFLE:
            # 変数に*を前置するとpositional argumentに変更可
            labels_, fvs_ = util.shuffle(*train_data)
        else:
            labels_, fvs_ = train_data
        
        for epoch in range(NUM_EPOCS):
            for i, (label_, fv_) in enumerate(zip(labels_, fvs_)):
                feed = {signed_label:label_, indices:fv_}
                _, cur_entropy, summary = sess.run([train_op, cross_entropy, merged], feed_dict=feed)
                if epoch == 0 and i % 10 == 0:
                    train_writer.add_summary(summary, global_step=i)

                print("epoch:{}\ttrain_data:{}\tcross_entropy:{}".format(epoch, i, cur_entropy))
        print("--- training finished ---")

        ### Evaluation ###

        # 初期化処理(local_variableのみ)
        eval_init_op = tf.local_variables_initializer()
        sess.run(eval_init_op)

        print("--- evaluation ---")        
        labels_, fvs_ = test_data
        for i, (label_, fv_) in enumerate(zip(labels_, fvs_)):
            feed = {signed_label:label_, indices:fv_}
            acc, pre, rec = sess.run([
                accuracy_update_op,
                precision_update_op,
                recall_update_op
            ], feed_dict=feed)
            print("iter:{}\tacc:{}\tpre:{}\trec:{}".format(i, acc, pre, rec))
        print("accuracy:", acc)
        print("f-measure:", 2*(pre*rec)/(pre+rec))
