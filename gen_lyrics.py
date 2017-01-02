import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.ops import rnn_cell, seq2seq


class HParam():

    batch_size = 32
    n_epoch = 100
    learning_rate = 0.01
    decay_steps = 1000
    decay_rate = 0.9
    grad_clip = 5

    state_size = 100
    num_layers = 3
    seq_length = 20
    log_dir = './logs'
    metadata = 'metadata.tsv'
    gen_num = 500 # how many chars to generate


class DataGenerator():

    def __init__(self, datafiles, args):    #数据读取与预处理
        self.seq_length = args.seq_length
        self.batch_size = args.batch_size
        with open(datafiles, encoding='utf-8') as f:
            self.data = f.read()    #data包含所有的数据，包括空格、字母、符号、汉字
        self.total_len = len(self.data)  # 起始样本的长度大约为60000多
        self.words = list(set(self.data))   
        self.words.sort()
        # vocabulary
        self.vocab_size = len(self.words)  #去除相同的，一共有2600多
        print('Vocabulary Size: ', self.vocab_size)
        self.char2id_dict = {w: i for i, w in enumerate(self.words)}    #在2600多个中根据字来获取对应的编号
        self.id2char_dict = {i: w for i, w in enumerate(self.words)}    #在2600多个中根据编号来获取对应的字
        # pointer position to generate current batch
        self._pointer = 0

        # save metadata file
        self.save_metadata(args.metadata)   #存储2600多在一个表中
    def char2id(self, c):
        return self.char2id_dict[c]

    def id2char(self, id):
        return self.id2char_dict[id]

    def save_metadata(self, file):
        with open(file, 'w') as f:
            f.write('id\tchar\n')
            for i in range(self.vocab_size):
                c = self.id2char(i)
                f.write('{}\t{}\n'.format(i, c))

    def next_batch(self):   #每次取一个batch_size=32，每个的维度是20
        x_batches = []
        y_batches = []
        for i in range(self.batch_size):    #对于32个x的每一个x，都是按照以下方法获取
            if self._pointer + self.seq_length + 1 >= self.total_len:
                self._pointer = 0
            bx = self.data[self._pointer: self._pointer + self.seq_length]  #在60000多个字中，取第一个字和他后面的19个字，一个20个字
            by = self.data[self._pointer +
                           1: self._pointer + self.seq_length + 1]  #相对于bx，向后移一位
            self._pointer += self.seq_length  # update pointer position

            # convert to ids
            bx = [self.char2id(c) for c in bx]  #把该20个字按照words转换成一个维度为20的向量，下同
            by = [self.char2id(c) for c in by]
            x_batches.append(bx)
            y_batches.append(by)

        return x_batches, y_batches #返回了32*20（即[[],[],[],[],[],... ...[]]），32是批处理的量，20是输入的维度
class Model():  #创建模型，训练或者测试
    """
    The core recurrent neural network model.
    """

    def __init__(self, args, data, infer=False):
        if infer:   #只有测试才用
            args.batch_size = 1
            args.seq_length = 1
        with tf.name_scope('inputs'):
            self.input_data = tf.placeholder(
                tf.int32, [args.batch_size, args.seq_length])   #输入占位符，实际输入为x_batches
            self.target_data = tf.placeholder(
                tf.int32, [args.batch_size, args.seq_length])   #输出占位符，相当于labels,实际为y_batches
            
        with tf.name_scope('model'):
            self.cell = rnn_cell.BasicLSTMCell(args.state_size) #创建cell，即该隐藏层有100个结点
            self.cell = rnn_cell.MultiRNNCell([self.cell] * args.num_layers)    #一共有三个cell，
            self.initial_state = self.cell.zero_state(
                args.batch_size, tf.float32)
            with tf.variable_scope('rnnlm'):    #softmax参数w、b
                w = tf.get_variable(
                    'softmax_w', [args.state_size, data.vocab_size])    #将结果对应到2600多个字中，即要分成2600多类
                b = tf.get_variable('softmax_b', [data.vocab_size])
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable(
                        'embedding', [data.vocab_size, args.state_size])    #2600多*100，即每个词都是一个100维的向量
                    inputs = tf.nn.embedding_lookup(embedding, self.input_data) #将input_data中的每个词表示成100维的词向量，inputs=32*20*100
            outputs, last_state = tf.nn.dynamic_rnn(
                self.cell, inputs, initial_state=self.initial_state) #将imputs放入网络，得到结果outputs，与每个cell的c、h参数，每个c=h=32*100
        with tf.name_scope('loss'):
            output = tf.reshape(outputs, [-1, args.state_size])#将输出的结果outputs=32*20*100，reshape为640*100的矩阵
            
            self.logits = tf.matmul(output, w) + b  #将输出结果分类到2600多个类中去
            self.probs = tf.nn.softmax(self.logits)
            self.last_state = last_state    #保留三个cell的所有参数
            
            targets = tf.reshape(self.target_data, [-1])    #为了方便计算，将target，reshap成output一样的矩阵
            loss = seq2seq.sequence_loss_by_example([self.logits],  #计算loss
                                                    [targets],
                                                    [tf.ones_like(targets, dtype=tf.float32)])
            self.cost = tf.reduce_sum(loss) / args.batch_size   #计算总loss
            tf.scalar_summary('loss', self.cost)    #带summary的应该都是画图，暂时不管
            
        with tf.name_scope('optimize'):
            self.lr = tf.placeholder(tf.float32, [])
            tf.scalar_summary('learning_rate', self.lr)

            optimizer = tf.train.AdamOptimizer(self.lr) #使用adam优化器进行优化
            tvars = tf.trainable_variables()    #返回所有变量
            grads = tf.gradients(self.cost, tvars)
            for g in grads:
                tf.histogram_summary(g.name, g)
            grads, _ = tf.clip_by_global_norm(grads, args.grad_clip)    #求grads要使用clip避免梯度爆炸，这里设置的阈值是5
            
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))    #将计算的梯度应用到变量上，对变量进行更新
            self.merged_op = tf.merge_all_summaries()


def train(data, model, args):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) #sess.run
        saver = tf.train.Saver()
        writer = tf.train.SummaryWriter(args.log_dir, sess.graph)

        # Add embedding tensorboard visualization. Need tensorflow version
        # >= 0.12.0RC0
        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'rnnlm/embedding:0'
        embed.metadata_path = args.metadata
        projector.visualize_embeddings(writer, config)

        max_iter = args.n_epoch * \ #迭代次数
            (data.total_len // args.seq_length) // args.batch_size
        for i in range(max_iter):
            learning_rate = args.learning_rate * \  #rate
                (args.decay_rate ** (i // args.decay_steps))
            x_batch, y_batch = data.next_batch()    #x_batch=32*20=y_batch
            feed_dict = {model.input_data: x_batch,
                         model.target_data: y_batch, model.lr: learning_rate}   #喂数
            train_loss, summary, _, _ = sess.run([model.cost, model.merged_op, model.last_state, model.train_op],
                                                 feed_dict) #迭代一次后计算的loss
            
            if i % 10 == 0:
                writer.add_summary(summary, global_step=i)
                print('Step:{}/{}, training_loss:{:4f}'.format(i,
                                                               max_iter, train_loss))
            if i % 2000 == 0 or (i + 1) == max_iter:    #保存模型
                saver.save(sess, os.path.join(
                    args.log_dir, 'lyrics_model.ckpt'), global_step=i)


def sample(data, model, args):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(args.log_dir)
        saver.restore(sess, ckpt)

        # initial phrase to warm RNN
        prime = u'你要离开我知道很简单'
        state = sess.run(model.cell.zero_state(1, tf.float32))

        for word in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = data.char2id(word)
            feed = {model.input_data: x, model.initial_state: state}
            state = sess.run(model.last_state, feed)

        word = prime[-1]
        lyrics = prime
        for i in range(args.gen_num):   #产生500个字的歌
            x = np.zeros([1, 1])
            x[0, 0] = data.char2id(word)
            feed_dict = {model.input_data: x, model.initial_state: state}
            probs, state = sess.run([model.probs, model.last_state], feed_dict)
            p = probs[0]
            word = data.id2char(np.argmax(p))
            print(word, end='')
            sys.stdout.flush()
            time.sleep(0.05)
            lyrics += word
        return lyrics


def main(infer):

    args = HParam()
    data = DataGenerator('JayLyrics.txt', args)
    model = Model(args, data, infer=infer)

    run_fn = sample if infer else train

    run_fn(data, model, args)


if __name__ == '__main__':
    msg = """
    Usage:
    Training: 
        python3 gen_lyrics.py 0
    Sampling:
        python3 gen_lyrics.py 1
    """
    if len(sys.argv) == 2:
        infer = int(sys.argv[-1])
        print('--Sampling--' if infer else '--Training--')
        main(infer)
    else:
        print(msg)
        sys.exit(1)
