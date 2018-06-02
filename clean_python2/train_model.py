from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib2, json
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
import os

class train_model(object):
	def __init__(self, day_unroll, week_unroll, batch_size, output_norm_dict, num_nodes, learning_rate, min_learning_rate, day_feature, week_feature, n_hot=2):
		self.global_steps =tf.Variable(0,name='global_steps',trainable=False)
		self.dropout = tf.placeholder (tf.float32, name='dropout')
		self.batch_size = batch_size
		self.n_hot = n_hot
		self.day_unroll = day_unroll
		self.week_unroll= week_unroll
		self.train_inputs_day = tf.placeholder(tf.float32, shape=[self.batch_size, day_unroll, day_feature],name='train_dayinputs')
		self.train_inputs_week= tf.placeholder(tf.float32, shape=[self.batch_size, week_unroll, week_feature],name='train_weekinputs')
		self.learning_rate= tf.Variable(learning_rate,name='tf_learning_rate')
		self.label_sz =  output_norm_dict.shape[0]
		self.label = tf.placeholder(tf.float32, shape=[batch_size, output_norm_dict.shape[0], n_hot], name = 'train_outputs')
		self.tf_learning_rate =     tf.Variable(learning_rate,name='tf_learning_rate')
		self.tf_min_learning_rate = tf.Variable(min_learning_rate,dtype=tf.float32,name='tf_min_learning_rate')
		self.num_nodes = num_nodes
		
		self._build_model()
		self._compute_loss()
		self._optimizer()
		#self.is_train = tf.placeholder (tf.bool, name='is_train')
    
	def _build_model(self):
        
		day_lstm_cells = [
		    tf.contrib.rnn.LSTMCell(num_units=self.num_nodes[li],
		                            state_is_tuple=True,
		                            initializer= tf.contrib.layers.xavier_initializer()
		                           )
		 for li in range(len(self.num_nodes))]

		week_lstm_cells = [
		    tf.contrib.rnn.LSTMCell(num_units=self.num_nodes[li],
		                            state_is_tuple=True,
		                            initializer= tf.contrib.layers.xavier_initializer()
		                           )
		 for li in range(len(self.num_nodes))]

		drop_day_lstm_cells = [tf.contrib.rnn.DropoutWrapper(
		        lstm, input_keep_prob=1.0,output_keep_prob=1.0-self.dropout, state_keep_prob=1.0-self.dropout
		) for lstm in day_lstm_cells]
		drop_week_lstm_cells = [tf.contrib.rnn.DropoutWrapper(
		        lstm, input_keep_prob=1.0,output_keep_prob=1.0-self.dropout, state_keep_prob=1.0-self.dropout
		) for lstm in week_lstm_cells]

		drop_day_multi_cell      = tf.contrib.rnn.MultiRNNCell(drop_day_lstm_cells)
		drop_week_multi_cell     = tf.contrib.rnn.MultiRNNCell(drop_week_lstm_cells)
		self.initial_state_day        = drop_day_multi_cell.zero_state(self.batch_size, tf.float32)
		self.initial_state_week       = drop_week_multi_cell.zero_state(self.batch_size, tf.float32)

		#outputs_day shape=[batch_size, max_time, output feature size]
		outputs_day,  final_state_day  = tf.nn.dynamic_rnn(drop_day_multi_cell, self.train_inputs_day,time_major=False, dtype=tf.float32, scope='lstm_day', initial_state = self.initial_state_day)
		outputs_week, final_state_week = tf.nn.dynamic_rnn(drop_week_multi_cell, self.train_inputs_week,time_major=False,dtype=tf.float32, scope='lstm_week', initial_state=self.initial_state_week)
		
		if len(outputs_day[: ,-1,:].get_shape().as_list()) <3:
			outputs_day =tf.expand_dims(outputs_day[: ,-1,:],1)
			outputs_week =tf.expand_dims(outputs_week[: ,-1,:],1)
		
		self.output_blend =  tf.concat((outputs_day, outputs_week), axis=2)
		#print (self.output_blend.get_shape().as_list())
		self.output_blend = tf.squeeze(self.output_blend,[1])
		
	def _compute_loss(self):
		self.predictions_pre = tf.contrib.layers.fully_connected(self.output_blend, 500, activation_fn=tf.nn.selu)
		self.predictions= tf.contrib.layers.fully_connected(self.predictions_pre, self.label_sz*self.n_hot, activation_fn=None)
		self.predictions = tf.reshape(self.predictions, shape=[-1,self.label_sz, self.n_hot])
		self.predictions_softmax = tf.nn.softmax(self.predictions)
		self.cost        = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.predictions))

	def _optimizer(self):
		learning_rate = tf.maximum(tf.train.exponential_decay(\
		self.tf_learning_rate, self.global_steps, decay_steps=1, decay_rate=0.8, staircase=True),self.tf_min_learning_rate)

		optimizer = tf.train.AdamOptimizer(learning_rate)
		gradients, v = zip(*optimizer.compute_gradients(self.cost))
		gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
		self.optimizer = optimizer.apply_gradients(
		    zip(gradients, v),  global_step = self.global_steps)
