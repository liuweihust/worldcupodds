#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

from nets import nets_factory
from datasets import WorldCup
from deployment import model_deploy
from tensorflow.contrib import slim as slim
#from tensorflow.train import Saver

tf.app.flags.DEFINE_string('model_name', 'WinLossNet', 'The name of the architecture to train.')
tf.app.flags.DEFINE_integer("training_epochs",10000,'Training epochs')
tf.app.flags.DEFINE_float('learning_rate', 0.0001,'learning rate')
tf.app.flags.DEFINE_integer('log_every_n_steps', 10,'')

tf.app.flags.DEFINE_string('dataset_name','WorldCup','')
tf.app.flags.DEFINE_integer('batch_size', 20, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('max_number_of_steps', 1000, 'The number of samples in each batch.')
tf.app.flags.DEFINE_string('train_path','/Users/liuwei/Documents/svn/worldcupodds/train','')
tf.app.flags.DEFINE_string('dataset_dir','/Users/liuwei/Documents/svn/worldcupodds/data/','')
tf.app.flags.DEFINE_integer('num_clones', 1,'Number of model clones to deploy.')
tf.app.flags.DEFINE_boolean('clone_on_cpu', True,'Use CPUs to deploy clones.')
FLAGS = tf.app.flags.FLAGS

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def main(_):
    dataset = WorldCup.dataset_worldcup()
    dataset.loaddata(FLAGS.dataset_dir)
    #dataset.loaddata()

    input_shape=dataset.GetInputShape()
    ckptfile=FLAGS.train_path+'/model.ckpt'

    tf.logging.set_verbosity(tf.logging.DEBUG)

    WinLossNet = nets_factory.get_network(FLAGS.model_name)
    WinLossNetLoss = nets_factory.get_loss(FLAGS.model_name)
    with tf.Graph().as_default():
        global_step = tf.train.create_global_step()

        X = tf.placeholder(tf.float32, [None, input_shape],name="input")
        Y = tf.placeholder(tf.int32, [None,],name="Y")

        logits = WinLossNet(X)
        loss = WinLossNetLoss(logits,Y)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            saver = tf.train.Saver()

            writer = tf.summary.FileWriter(FLAGS.train_path, sess.graph)
            summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
            for loss in tf.get_collection(tf.GraphKeys.LOSSES):
                #summaries.add(tf.summary.scalar(loss.op.name, loss))
                variable_summaries(loss)

            merged = tf.summary.merge_all()

            #saver.restore(sess,ckptfile)
            for epoch in range(FLAGS.max_number_of_steps):
                Inx,Iny = dataset.GetNextbatch()
                print("In",Inx,Iny)
                _, c = sess.run( [optimizer, loss], 
                            feed_dict={X:Inx, Y:np.array(Iny, dtype=np.int32)} )
                if epoch % FLAGS.log_every_n_steps == 0:
                    print("%d:Test cost:%f"%(epoch,c))

                    save_path = saver.save(sess,ckptfile)
                    print("Model saved in file: %s" % save_path)

            writer.close()
if __name__ == '__main__':
    tf.app.run()
