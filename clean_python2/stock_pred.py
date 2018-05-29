from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib2, json
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
import os
from DataGeneratorSeq import DataGeneratorSeq
from train_model import train_model

def download_preprocess_data(config):
	url_string_day  = \
	"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(config.ticker,config.api_key)
	url_string_week = \
	"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=%s&outputsize=full&apikey=%s"%(config.ticker,config.api_key)
	    # Save data to this file
	file_to_save_day  = 'stock_market_data_day-%s.csv'%config.ticker
	file_to_save_week = 'stock_market_data_week-%s.csv'%config.ticker
	    # If you haven't already saved data,
	    # Go ahead and grab the data from the url
	    # And store date, low, high, volume, close, open values to a Pandas DataFrame
	if (not os.path.exists(file_to_save_day)) or (not os.path.exists(file_to_save_week)):
	    url_day = urllib2.urlopen(url_string_day)
	    data_day = json.loads(url_day.read().decode())
	    
	    url_week = urllib2.urlopen(url_string_week)
	    data_week = json.loads(url_week.read().decode())
	    # extract stock market data
	    data_day = data_day['Time Series (Daily)']
	    data_week = data_week['Weekly Time Series']
	    df_day  = pd.DataFrame(columns=['Date','Mid' ,'Volume'])
	    df_week = pd.DataFrame(columns=['Date','Mid' ,'Volume'])
	    for k,v in data_day.items():
		date = dt.datetime.strptime(k, '%Y-%m-%d')
		data_row = [date.date(),(float(v['3. low'])+float(v['2. high']))/2, float(v['5. volume'])]
		df_day.loc[-1,:] = data_row
		df_day.index = df_day.index + 1
	    print('Data saved to : %s'%file_to_save_day)
	    df_day = df_day.sort_values('Date')
	    df_day.to_csv(file_to_save_day)
		
	    for k,v in data_week.items():
		date = dt.datetime.strptime(k, '%Y-%m-%d')
		data_row = [date.date(),(float(v['3. low'])+float(v['2. high']))/2, float(v['5. volume'])]
		df_week.loc[-1,:] = data_row
		df_week.index = df_week.index + 1
	    print('Data saved to : %s'%file_to_save_week)
	    df_week = df_week.sort_values('Date')
	    df_week.to_csv(file_to_save_week)

	    # If the data is already there, just load it from the CSV
	else:
		print('File already exists. Loading data from CSV')
		df_day = pd.read_csv(file_to_save_day)
		df_week = pd.read_csv(file_to_save_week)

	pre_train_data_day  =  df_day.loc[:,['Mid','Volume']].as_matrix()
	pre_train_data_week =  df_week.loc[:,['Mid','Volume']].as_matrix()

	#align pre_train_data_data with data week
	align=0
	while(df_day['Date'].loc[align] <=df_week['Date'].loc[0]):
	    align=align+1

	pre_train_data_day = pre_train_data_day[align:]
	    
	total_day_1y = 52*5
	total_week_1y =52-9

	train_data_day  = np.zeros(( pre_train_data_day.shape[0] - total_day_1y,2))
	config.day_week_map    = np.zeros(( pre_train_data_day.shape[0] - total_day_1y)) #map day to corresponing week data
	j=0
	for i in range(0, train_data_day.shape[0]): 
	    while ( df_day['Date'].loc[i+align+total_day_1y] >df_week['Date'].loc[j+total_week_1y]):
		j= j+1
	    config.day_week_map[i] =j-1
	train_data_week = np.zeros(( pre_train_data_week.shape[0]-total_week_1y,2))



	#discard first year data for moving average
	average_day = np.mean( pre_train_data_day, axis=0) 
	average_week = np.mean( pre_train_data_week, axis=0) 


	EMA = np.array(0)
	for i in range(0, train_data_day.shape[0]):
	    average_day = average_day+ (pre_train_data_day[i+total_day_1y]- pre_train_data_day[i])/ total_day_1y
	    train_data_day[i,0] =  (pre_train_data_day[i+total_day_1y,0]- pre_train_data_day[i-1+total_day_1y,0])/pre_train_data_day[i-1+total_day_1y,0]
	    train_data_day[i,1:] = (pre_train_data_day[i+total_week_1y,1:]/(average_day[1:]))*0.95+  EMA*0.05
	    #train_data_day[i] = (pre_train_data_day[i+total_day_1y]) *0.95 + EMA*0.05
	    EMA = train_data_day[i, 1:]
	for i in range(0,train_data_week.shape[0]):
	    average_week = average_week+ (pre_train_data_week[i+total_week_1y]- pre_train_data_week[i])/ total_week_1y
	    train_data_week[i,0] =  (pre_train_data_week[i+total_week_1y,0]- pre_train_data_week[i-1+total_week_1y,0])/pre_train_data_week[i-1+total_week_1y,0]
	    train_data_week[i,1:] = (pre_train_data_week[i+total_week_1y,1:]/(average_week[1:]))*0.95+  EMA*0.05
	    EMA = train_data_week[i,1:]

	    
	config.average_train_day = np.amax(train_data_day, axis=0)
	config.average_train_week = np.amax(train_data_week, axis=0)
	config.train_data_day = train_data_day/ config.average_train_day
	config.train_data_week = train_data_week/ config.average_train_week
	config.ori_data_day = pre_train_data_day[total_day_1y:]
	config.ori_data_week= pre_train_data_week[total_week_1y:]

### define train one step procedure
def train_one_step(sess,iteration, dg, tm):
    
    day_input, week_input, output_label = dg.onestep_unroll()
    feed = {tm.train_inputs_day: day_input, tm.train_inputs_week: week_input, tm.label:output_label, tm.dropout:0.2}
    _, step_cost, predictions = sess.run([tm.optimizer, tm.cost, tm.predictions], feed_dict=feed)
    correct = 1*np.equal( np.argmax(predictions, axis=2), np.argmax(output_label, axis=2)) 
    train_acc = np.mean( correct)
    
    if tm.n_hot>2:
        condense_predictions = np.zeros(( predictions.shape[0],  predictions.shape[1],  2))
        condense_predictions[:,:,0] = predictions[ :,:,0]
        condense_predictions[:,:,1] = predictions[ :,:,predictions.shape[2]//2]
        condense_label = np.zeros( (output_label.shape[0],  output_label.shape[1],  2))
        condense_label[:,:,0] = output_label[ :,:,0]
        condense_label[:,:,1] = output_label[ :,:,predictions.shape[2]//2]
        for i in range(1,predictions.shape[2]//2):
            condense_predictions[:,:,0] = condense_predictions[:,:,0]+ predictions[ :,:,i]
            condense_predictions[:,:,1] = condense_predictions[:,:,1]+ predictions[ :,:,i+predictions.shape[2]//2 ]
            condense_label[:,:,0] = condense_label[:,:,0]+ output_label[ :,:,i]
            condense_label[:,:,1] = condense_label[:,:,1]+ output_label[ :,:,i+predictions.shape[2]//2 ]
        
        condense_correct = 1*np.equal( np.argmax( condense_predictions, axis=2), np.argmax(condense_label, axis=2)) 
        condense_train_acc = np.mean(condense_correct)
    #print 'step:%d , cost:%f' % (iteration, step_cost)
    
    if (iteration%10 ==1):
        day_input, week_input, output_label = dg.onestep_unroll(test=True)
        feed = {tm.train_inputs_day: day_input, tm.train_inputs_week: week_input, tm.label:output_label, tm.dropout:0}
        test_cost, predictions = sess.run([tm.cost, tm.predictions], feed_dict=feed)
        correct = 1*np.equal( np.argmax(predictions, axis=2), np.argmax(output_label, axis=2)) 
        test_acc = np.mean( correct)
        if tm.n_hot>2:
            condense_predictions = np.zeros(( predictions.shape[0],  predictions.shape[1],  2))
            condense_predictions[:,:,0] = predictions[ :,:,0]
            condense_predictions[:,:,1] = predictions[ :,:,predictions.shape[2]//2]
            condense_label = np.zeros( ( output_label.shape[0],  output_label.shape[1],  2))
            condense_label[:,:,0] = output_label[ :,:,0]
            condense_label[:,:,1] = output_label[ :,:,predictions.shape[2]//2]
            for i in range(1,predictions.shape[2]//2):
                condense_predictions[:,:,0] = condense_predictions[:,:,0]+ predictions[ :,:,i]
                condense_predictions[:,:,1] = condense_predictions[:,:,1]+ predictions[ :,:,i+predictions.shape[2]//2 ]
                condense_label[:,:,0] = condense_label[:,:,0]+ output_label[ :,:,i]
                condense_label[:,:,1] = condense_label[:,:,1]+ output_label[ :,:,i+predictions.shape[2]//2 ]
            condense_correct = 1*np.equal( np.argmax( condense_predictions, axis=2), np.argmax(condense_label, axis=2)) 
            condense_test_acc = np.mean(condense_correct)
            print 'step:%d , train_acc:%f, train_acc_s:%f, test_acc:%f, test_acc_s:%f' % (iteration, train_acc, condense_train_acc, test_acc, condense_test_acc)
        else:
            print 'step:%d , train_acc:%f, test_acc:%f ' % (iteration, train_acc, test_acc)
    return step_cost

### define test procedure
def test_all(sess, dg, tm):
    #overall test
    all_predictions = np.zeros( (1, 1,tm.n_hot))
    all_output = np.zeros((1, 1, tm.n_hot))
    for i in range( dg._day_data.shape[0]-dg.test_sz,dg._day_data.shape[0]-21-tm.batch_size,tm.batch_size):
        day_input, week_input, output_label = dg.run_all_test(i)
        feed = {tm.train_inputs_day: day_input, tm.train_inputs_week: week_input, tm.label:output_label, tm.dropout:0}
        predictions = sess.run([tm.predictions], feed_dict=feed)[0]
        all_predictions=np.append(all_predictions,predictions, axis=0)
        all_output=     np.append(all_output,output_label, axis=0)
    
    
    all_correct = 1*np.equal( np.argmax(all_predictions, axis=2), np.argmax(all_output, axis=2)) 
    train_acc = np.mean( all_correct)
    print 'test overall predictions:%s' % (train_acc)

def train(config, dg):
	session_config = tf.ConfigProto(
                allow_soft_placement=True,
                gpu_options=tf.GPUOptions(allow_growth=True),
                device_count={'GPU': 1})

	tf.reset_default_graph()
	sess=tf.Session(config=session_config)
	if config.predict_type=='binary': n_hot=2
	if config.predict_type=='4class': n_hot=4

	tm =  train_model(config.day_unroll, config.week_unroll, config.bz_size, config.output_normalize_dict, config.num_nodes, config.learning_rate, config.min_learning_rate,\
 config.train_data_day.shape[1], config.train_data_week.shape[1], n_hot=n_hot)
	check_op = tf.no_op()
	sess.run(tf.global_variables_initializer())    
	saver = tf.train.Saver()
	for i in range(config.iterations):
	    train_one_step(sess,i,dg,tm)
	test_all(sess, dg, tm)
	saver.save(sess, "checkpoints/model_"+config.predict_type+"_"+str(int(config.output_normalize_dict[0,0]))+"_.ckpt")
	sess.close()
	
def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--api_key',    type=str, default='J0LLMPESJ7H0SKXQ')
	parser.add_argument('--ticker',      type=str, default='SPX')
	parser.add_argument('--iterations'   , type=int, default=1000)
	parser.add_argument('--test_sz' ,    type=int, default=500)
	parser.add_argument('--bz_size' ,     type=int, default=128)       
	parser.add_argument('--day_unroll',   type=int, default=20)
	parser.add_argument('--week_unroll',  type=int, default=12)
	parser.add_argument('--predict_type', type=str, default='binary', choices=['binary', '4class'])	
	parser.add_argument('--day',          type=int, default=1, choices=[1, 5, 20]) #1day, 1week, 1m	
	parser.add_argument('--mode',         type=str, default='train')

                                    
	config = parser.parse_args()	
	#day                #1week              #4week
	if config.day ==1:
		config.output_normalize_dict = np.array([ [1,0.002,0.0085]])
	elif config.day==5:
		config.output_normalize_dict = np.array([ [5,0.004,0.00191]])
	elif config.day==20:
		config.output_normalize_dict = np.array([ [20,0.0157,0.038]])
                                   
	############# LSTM model architecture
	#lstm with 3 layers each with dimension as followed
	config.num_nodes = [300,200,250]
	config.n_layers = len(config.num_nodes) # number of layers
	config.dropout = 0.2 # dropout amount
	#########learning paramters
	config.learning_rate= 0.01
	config.min_learning_rate= 0.001
	
	download_preprocess_data(config) # download and preprocess data
	dg = DataGeneratorSeq(config.train_data_day, config.train_data_week, config.bz_size , \
                 config.day_unroll, config.week_unroll,config.output_normalize_dict, config.day_week_map, config.test_sz,  config.ori_data_day, config.ori_data_week,\
                  predict_type=config.predict_type)

	if config.mode == 'train':
		train(config, dg)
	
if __name__ == '__main__':
		main()


