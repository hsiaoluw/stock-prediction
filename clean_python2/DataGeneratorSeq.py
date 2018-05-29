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
                 num_week_unroll,output_norm_dict, day_week_map, test_sz, ori_data_day, ori_data_week, predict_type ='binary'):
        self._day_data = day_data
        self._week_data = week_data
        self._day_week_map= day_week_map
        self._output_norm_dict = output_norm_dict
        self._day_length = len(self._day_data) - num_day_unroll - output_norm_dict[output_norm_dict.shape[0]-1,0]*5-test_sz
        self._batch_size = batch_size
        self._num_week_unroll = num_week_unroll
        self._num_day_unroll = num_day_unroll
        self._segments = self._day_length //self._batch_size
        self._cursor = [(offset+1) * self._segments-1 for offset in range(self._batch_size)]
        self._test_cursor =  np.random.randint( self._day_length+num_day_unroll,\
                                               len(self._day_data)-1-output_norm_dict[output_norm_dict.shape[0]-1,0]*5, size=[self._batch_size])
        self.predict_type = predict_type
        self.ori_data_day  = ori_data_day # non normalize price
        self.ori_data_week = ori_data_week #non normalize price
        self.test_sz = test_sz

    def _get_labels(self, test= False):
        day_data = self.ori_data_day
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
                    result =  (day_data[int(cursor[b]+self._output_norm_dict[0,0]),0]- 
                                                  day_data[int(cursor[b]),0])/day_data[int(cursor[b]),0]
                if self.predict_type=='binary': batch_labels[b,0,:]=(np.array([0,1]) if result>=0 else np.array([1,0]))   
                if self.predict_type=='4class':
                    batch_labels[b,j]  = np.zeros((4), dtype=np.float32)
                    if  (result>self._output_norm_dict[j,1]):     batch_labels[b,j,3]=1  
                    elif(result>0):                               batch_labels[b,j,2]=1  
                    elif(result<-1*self._output_norm_dict[j,1]):  batch_labels[b,j,0]=1 
                    else:  batch_labels[b,j,1]=1  
                    
               
        #print self._output_norm_dict[0,0], self._cursor[0]+self._output_norm_dict[0,0], batch_labels[0,0]
       
        return batch_labels
    def _get_onestep_weekbatch(self, i, test=False):
        cursor = self._test_cursor if(test) else self._cursor
        week_batch_data = np.zeros((self._batch_size, self._day_data.shape[1]),dtype=np.float32)
        for b in range(self._batch_size):
            week_batch_data[b] = self._week_data[int(self._day_week_map[int(cursor[b])]-i)]
        return week_batch_data
    def _get_onestep_daybatch(self,i, test=False):
        cursor = self._test_cursor if(test) else self._cursor
        day_batch_data = np.zeros((self._batch_size, self._day_data.shape[1]),dtype=np.float32)
        for b in range(self._batch_size):
            k=int(cursor[b]-i)
            day_batch_data[b] = self._day_data[k]
        return day_batch_data
   

    def onestep_unroll(self, test=False):

        unroll_weekdata,unroll_daydata = [],[]
        for i in range(self._num_day_unroll):
            unroll_daydata.append(self._get_onestep_daybatch(self._num_day_unroll-1-i, test))
        for i in range(self._num_week_unroll):
            unroll_weekdata.append(self._get_onestep_weekbatch(self._num_week_unroll-i, test))

        onestep_label=self._get_labels(test)
        #update label
        self.reset_indices(test)
        unroll_daydata  = np.transpose( unroll_daydata,  (1, 0, 2))
        unroll_weekdata = np.transpose( unroll_weekdata, (1, 0, 2))
        return unroll_daydata, unroll_weekdata, onestep_label

    def reset_indices(self, test=False):
        if(not test):
            for b in range(self._batch_size):
                self._cursor[b] = \
                np.random.randint( max((b) * self._segments, self._num_day_unroll), (b+1)*self._segments-1)
        else:
            self._test_cursor =  np.random.randint( self._day_length+self._num_day_unroll,\
                                               len(self._day_data)-1-self._output_norm_dict[self._output_norm_dict.shape[0]-1,0]*5, size=[self._batch_size])
    
    def run_all_test(self, i,test=True):
        # i the start of tests
        self._test_cursor = np.arange(i, i+self._batch_size)
        a,b,c = self.onestep_unroll(test=True)
        return a, b, c
