from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib2, json
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
import os

class DataGeneratorSeq(object):

	def __init__(self,day_data, week_data,batch_size,num_day_unroll,\
                 num_week_unroll,output_norm_dict, day_week_map, test_sz, ori_data_day, ori_data_week, five_day_avg, five_day_avg_norm,predict_type ='binary'):
		self._day_data = day_data
		self._week_data = week_data
		self._day_week_map= day_week_map
		self._output_norm_dict = output_norm_dict
		self._day_length = len(self._day_data) - max(num_day_unroll, num_week_unroll*5) - output_norm_dict[output_norm_dict.shape[0]-1,0]*5-test_sz
		self._batch_size = batch_size
		self._num_week_unroll = num_week_unroll
		self._num_day_unroll = num_day_unroll
		self._segments = self._day_length //self._batch_size
		self._cursor = [(offset+1) * self._segments-1+ num_week_unroll*5 for offset in range(self._batch_size)]
		self._test_cursor =  np.random.randint( self._day_length,\
		                                       len(self._day_data)-1-output_norm_dict[output_norm_dict.shape[0]-1,0]*5, size=[self._batch_size])
		self.predict_type = predict_type
		self.ori_data_day  = ori_data_day # non normalize price
		self.ori_data_week = ori_data_week #non normalize price
		self.five_day_avg = five_day_avg
		self.test_sz = test_sz
		self.five_day_avg_norm = five_day_avg_norm

	def _get_labels(self, test= False):
		day_data = self.ori_data_day #i self._output_norm_dict[0,0]==1 else self.five_day_avg
		week_data = self.ori_data_week
		cursor = self._test_cursor if(test) else self._cursor
		
		if self.predict_type == 'binary':
		    batch_labels = np.zeros((self._batch_size, self._output_norm_dict.shape[0], 2),dtype=np.float32)
		elif self.predict_type == '4class': 
		    batch_labels = np.zeros((self._batch_size, self._output_norm_dict.shape[0], 4),dtype=np.float32)
		    
		for b in range(self._batch_size):    
		    for j in range(0,self._output_norm_dict.shape[0]):
		        if not j==0:
				result=  \
		                (week_data[ int(self._day_week_map[int(cursor[b])]+self._output_norm_dict[j,0]),0]\
		                -week_data[ int(self._day_week_map[int(cursor[b])]),0])/week_data[ int(self._day_week_map[int(cursor[b])]),0]
		        else:
				result =  (day_data[int(cursor[b]+self._output_norm_dict[0,0]),0]- day_data[int(cursor[b]),0])/day_data[int(cursor[b]),0]
		        if self.predict_type=='binary': batch_labels[b,0,:]=(np.array([0,1]) if result>=0 else np.array([1,0]))   
		        if self.predict_type=='4class':
		            batch_labels[b,j]  = np.zeros((4), dtype=np.float32)
		            if  (result>self._output_norm_dict[j,1]):     
							batch_labels[b,j,3]=0.9
							batch_labels[b,j,2]=0.1
		            elif(result>0):                               
							batch_labels[b,j,2]=0.95
							batch_labels[b,j,3]=0.05
		            elif(result<-1*self._output_norm_dict[j,1]):  
							batch_labels[b,j,0]=0.9
							batch_labels[b,j,1]=0.1
		            else:  
							batch_labels[b,j,1]=0.95
							batch_labels[b,j,0]=0.05
       
		return batch_labels
	
	def _get_onestep_weekbatch(self, i, test=False):
		cursor = self._test_cursor if(test) else self._cursor
		week_batch_data = np.zeros((self._batch_size, self.five_day_avg_norm.shape[1]),dtype=np.float32)
		for b in range(self._batch_size):
			week_batch_data[b] = np.copy(self._week_data[int(self._day_week_map[int(cursor[b])]-i)]) if not (self._output_norm_dict[0,0]==1) else self.five_day_avg_norm[int(cursor[b])-(i-1)*5]
		return week_batch_data

	def _get_onestep_daybatch(self,i, test=False):
		cursor = self._test_cursor if(test) else self._cursor
		_day_data = self._day_data
		day_batch_data = np.zeros((self._batch_size, _day_data.shape[1]),dtype=np.float32)
		for b in range(self._batch_size):
			day_batch_data[b] = np.copy(_day_data[int(cursor[b]-i)])
		return day_batch_data
   

	def onestep_unroll(self, test=False):

		unroll_weekdata,unroll_daydata = [],[]
		for i in range(self._num_day_unroll):
		    unroll_daydata.append(self._get_onestep_daybatch(self._num_day_unroll-1-i, test=test))
		for i in range(self._num_week_unroll):
		    unroll_weekdata.append(self._get_onestep_weekbatch(self._num_week_unroll-i, test=test))

		onestep_label=self._get_labels(test=test)
		#update label
		self.reset_indices(test=test)
		unroll_daydata  = np.transpose( unroll_daydata,  (1, 0, 2))
		unroll_weekdata = np.transpose( unroll_weekdata, (1, 0, 2))
		return unroll_daydata, unroll_weekdata, onestep_label

	def reset_indices(self, test=False):
		if(not test):
		    for b in range(self._batch_size):
		        self._cursor[b] = \
		        np.random.randint( (b) * self._segments, (b+1)*self._segments-1 )+ self._num_week_unroll*5
		else:
		    self._test_cursor =  np.random.randint( self._day_length+self._num_day_unroll,\
		                                       len(self._day_data)-1-self._output_norm_dict[self._output_norm_dict.shape[0]-1,0]*5, size=[self._batch_size])
    
	def run_all_test(self, i,test=True):
		# i the start of tests
		self._test_cursor = np.arange(i, i+self._batch_size)
		return  self.onestep_unroll(test=True)
		
