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
	
	n_day_smooth=3
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
	    config.df_day = df_day = pd.DataFrame(columns=['Date','Mid' ,'Volume'])
	    df_week = pd.DataFrame(columns=['Date','Mid' ,'Volume'])
	    for k,v in data_day.items():
		date = dt.datetime.strptime(k, '%Y-%m-%d')
		data_row = [date.date(),(float(v['4. close'])), float(v['5. volume'])]
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
		config.df_day =df_day= pd.read_csv(file_to_save_day)
		config.df_week = df_week=pd.read_csv(file_to_save_week)

	pre_train_data_day  =  df_day.loc[:,['Mid','Volume']].as_matrix()
	pre_train_data_week =  df_week.loc[:,['Mid','Volume']].as_matrix()

	#align pre_train_data_data with data week
	config.align= align=0
	while(df_day['Date'].loc[align] <=df_week['Date'].loc[0]):
	    align=align+1
	config.align= align
	pre_train_data_day = pre_train_data_day[align:]
	    
	config.total_day_1y = total_day_1y= 52*5-config.week_unroll*5
	config.total_week_1y= total_week_1y =52-config.week_unroll

	train_data_day  = np.zeros(( pre_train_data_day.shape[0] - total_day_1y,2))
	train_data_week = np.zeros(( pre_train_data_week.shape[0]-total_week_1y,2))
	config.day_week_map    = np.zeros(( pre_train_data_day.shape[0] - total_day_1y)) #map day to corresponing week data
	j=0
	for i in range(0, train_data_day.shape[0]): 
		if train_data_week.shape[0]<= j:
			config.day_week_map[i]=j
		else:
			while ( df_day['Date'].loc[i+align+total_day_1y] >df_week['Date'].loc[j+total_week_1y]):	
				j= j+1
				if j>= train_data_week.shape[0]:
					config.day_week_map[i] =j 
					break
			if j< train_data_week.shape[0]:	
				config.day_week_map[i] =j-1
	

	#discard first year data for moving average
	average_day = np.mean( pre_train_data_day[:total_day_1y], axis=0) 
	average_week = np.mean( pre_train_data_week[:total_week_1y], axis=0) 
	config.ori_data_day = np.copy(pre_train_data_day[total_day_1y:])
	config.five_day_avg      = np.copy(pre_train_data_day)

	for i in range(5, config.five_day_avg.shape[0]):
		config.five_day_avg[i]= (pre_train_data_day[i]-pre_train_data_day[i-5])*0.2+ config.five_day_avg[i-1]
	config.ori_data_day[0] = (pre_train_data_day[total_day_1y]+pre_train_data_day[total_day_1y-1]+ pre_train_data_day[total_day_1y-2])/3
	for i in range( 1, config.ori_data_day.shape[0]):
		config.ori_data_day[i]= (pre_train_data_day[i+total_day_1y]-pre_train_data_day[i+total_day_1y-config.n_day_smooth])*(1.0/config.n_day_smooth)+ config.ori_data_day[i-1]
		
	config.five_day_avg_norm = np.zeros(( config.five_day_avg.shape[0]-total_day_1y ,2))
	EMA = np.array(0)
	average_fiveday = average_day
	for i in range(0, train_data_day.shape[0]):
		average_day = average_day+ (pre_train_data_day[i+total_day_1y]- pre_train_data_day[i])/ total_day_1y
		train_data_day[i,0] =  (config.ori_data_day[i,0]- config.ori_data_day[i-1,0])/config.ori_data_day[i-1,0]
		train_data_day[i,1:] = (config.ori_data_day[i,1:]/(average_day[1:]))#*0.95+  EMA*0.05
		EMA = train_data_day[i, 1:]
	
	EMA = np.array(0)
	for i in range(0,train_data_week.shape[0]):
		average_week = average_week+ (pre_train_data_week[i+total_week_1y]- pre_train_data_week[i])/ total_week_1y
		train_data_week[i,0] =  (pre_train_data_week[i+total_week_1y,0]- pre_train_data_week[i-1+total_week_1y,0])/pre_train_data_week[i-1+total_week_1y,0]
		train_data_week[i,1:] = (pre_train_data_week[i+total_week_1y,1:]/(average_week[1:]))#*0.95+  EMA*0.05
		EMA = train_data_week[i,1:]
	EMA = np.array(0)	
	for i in range(0,config.five_day_avg_norm.shape[0]):
		average_fiveday = average_fiveday+ (config.five_day_avg[i+total_day_1y]- config.five_day_avg[i])/ total_day_1y
		config.five_day_avg_norm[i,0]  =   (config.five_day_avg[i+total_day_1y,0]- config.five_day_avg[i+total_day_1y-1*5,0])/ config.five_day_avg[i+total_day_1y-1*5,0]
		config.five_day_avg_norm[i,1:] =   (config.five_day_avg[i+total_day_1y,1:]/(average_fiveday[1:]))#*0.95+  EMA*0.05
		EMA = config.five_day_avg_norm[i,1:]

	    
	config.max_train_day =  config.train_data_max_day
	config.max_train_week = config.train_data_max_week
	config.train_data_day = train_data_day/ config.train_data_max_day
	#for i in range(config.train_data_day.shape[0]):
		#print '%s\t%f\t%f\t%f' %( config.df_day['Date'].loc[i+config.align+total_day_1y],config.train_data_day[i,0], config.ori_data_day[i,0], config.train_data_day[i,1])
	config.train_data_week = train_data_week/ config.max_train_week
	config.five_day_avg_norm = config.five_day_avg_norm/ config.max_train_day
	
	##macd
	config.macd= np.zeros((config.ori_data_day.shape[0], config.ori_data_day.shape[1]))
	#config.macd_five= np.zeros((config.ori_data_day.shape[0], config.ori_data_day.shape[1]))
	short=short_five=0
	long_=long_five=0
	for i in range(config.macd.shape[0]):
		short = config.ori_data_day[i]*0.15+ short*0.85
		long_  = config.ori_data_day[i]*0.075+ long_*0.925
		#short_five = config.five_day_avg[i+total_day_1y]*0.15+ short_five*0.85
		#long_five  = config.five_day_avg[i+total_day_1y]*0.075+ long_five*0.925
		config.macd[i]         = (short-long_)/config.ori_data_day[i]
		#config.macd_five[i] = (short_five-long_five)/config.five_day_avg[i+total_day_1y]
	config.train_data_day = np.concatenate( [config.train_data_day, config.macd], axis=1)
	#config.five_day_avg_norm = np.concatenate( [config.five_day_avg_norm, config.macd_five], axis=1)
	config.ori_data_week= np.copy(pre_train_data_week[total_week_1y:])
	
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
    
    if (iteration%100 ==1):
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
def test_all(sess, dg, tm, config):
    #overall test
	all_predictions = np.zeros( (1, 1,tm.n_hot))
	all_output = np.zeros((1, 1, tm.n_hot))
	for i in range( dg._day_data.shape[0]-dg.test_sz,dg._day_data.shape[0]-tm.batch_size-20,tm.batch_size):
		day_input, week_input, output_label = dg.run_all_test(i)
		feed = {tm.train_inputs_day: day_input, tm.train_inputs_week: week_input, tm.label:output_label, tm.dropout:0}
        	predictions = sess.run([tm.predictions_softmax], feed_dict=feed)[0]
		acc         = np.mean( 1*np.equal( np.argmax(predictions, axis=2), np.argmax(output_label, axis=2)) )
		print 'test from date %s to %s, with test acc=%f' % (  config.df_day['Date'].loc[config.align+i+ config.total_day_1y], config.df_day['Date'].loc[config.align+i+tm.batch_size-1+ config.total_day_1y], acc)
		#print output_label.shape
        	all_predictions=np.append(all_predictions,predictions, axis=0)
        	all_output=     np.append(all_output,output_label, axis=0)
    
	all_predictions = all_predictions[1:]
	#print all_predictions.shape
	all_output = all_output[1:]
	
	all_correct = 1*np.equal( np.argmax(all_predictions, axis=2), np.argmax(all_output, axis=2))
	max_output = np.argmax(all_output, axis=2)
	#for i in range(all_predictions.shape[0]):
		#if config.predict_type == '4class':
			#print '%s %f %f %f %f %d' % ( config.df_day['Date'].loc[config.align+i+ config.total_day_1y+ dg._day_data.shape[0]-dg.test_sz],all_predictions[i,0,0], all_predictions[i,0,1], all_predictions[i,0,2], all_predictions[i,0,3], max_output[i,0])
		#else:
			#print '%s %f %f %d' %       ( config.df_day['Date'].loc[config.align+i+ config.total_day_1y+ dg._day_data.shape[0]-dg.test_sz],all_predictions[i,0,0], all_predictions[i,0,1], max_output[i,0])
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
 config.train_data_day.shape[1], config.five_day_avg_norm.shape[1], n_hot=n_hot)
	check_op = tf.no_op()
	sess.run(tf.global_variables_initializer())    
	saver = tf.train.Saver()
	for i in range(config.iterations):
	    train_one_step(sess,i,dg,tm)
	test_all(sess, dg, tm, config)
	saver.save(sess, "checkpoints/model_"+config.predict_type+"_"+str(int(config.output_normalize_dict[0,0]))+"_.ckpt")
	sess.close()

def test(config):
	if get_testdata(config):
		session_config = tf.ConfigProto(
				allow_soft_placement=True,
		        	gpu_options=tf.GPUOptions(allow_growth=True),
		        	device_count={'GPU': 1})

		tf.reset_default_graph()
		sess=tf.Session(config=session_config)
		if config.predict_type=='binary': n_hot=2
		if config.predict_type=='4class': n_hot=4
		print "###predict_type:%s, num_days_after:%d" % (config.predict_type, config.day)
		tm =  train_model(config.day_unroll, config.week_unroll, 1, config.output_normalize_dict, config.num_nodes, config.learning_rate, config.min_learning_rate,\
 config.test_data_day.shape[1], config.five_day_avg_norm.shape[1], n_hot=n_hot)
		saver = tf.train.Saver()
		saver.restore(sess, config.ch_pt)
		
		all_predictions = np.zeros( (1, 1,tm.n_hot))
		all_output = np.zeros((1, 1, 1))
		ori =1.0
		nd = config.output_normalize_dict[0,0]
		ns = 1#config.n_day_smooth
		for i in range(config.week_unroll*5+1, config.test_data_day.shape[0]-20):
			test_data_day, test_data_week, test_output=test_onestep(config,i)
			feed = {tm.train_inputs_day: test_data_day, tm.train_inputs_week: test_data_week, tm.dropout:0}
			predictions = sess.run([tm.predictions_softmax], feed_dict=feed)[0]
			all_predictions = np.append(all_predictions, predictions, axis=0)
			all_output = np.append(all_output, test_output, axis=0)
			real_trend = ((config.df_day['Mid'].loc[config.align+i+nd+ config.total_day_1y]-config.df_day['Mid'].loc[config.align+i+ config.total_day_1y])/config.df_day['Mid'].loc[config.align+i+ config.total_day_1y])
			print real_trend, ori
			if predictions.shape[2] ==2: 
				print '%s\t%f\t%f\t%d' % ( config.df_day['Date'].loc[config.align+i+ config.total_day_1y], predictions[0,0,0], predictions[0,0,1], test_output[0,0,0])
				if predictions[0,0,1]>0.55: #and (config.df_day['Mid'].loc[config.align+i+config.total_day_1y]< config.df_day['Mid'].loc[config.align+i-1+ config.total_day_1y]):
					ori= ori*(1+real_trend)
				elif predictions[0,0,0]>0.55:  #and (config.df_day['Mid'].loc[config.align+i+config.total_day_1y]> config.df_day['Mid'].loc[config.align+i-1+ config.total_day_1y]) :
					ori= ori*(1-real_trend)
			else:
				print '%s\t%f\t%f\t%f\t%f\t%d' % ( config.df_day['Date'].loc[config.align+i+ config.total_day_1y], predictions[0,0,0], predictions[0,0,1], predictions[0,0,2], predictions[0,0,3], test_output[0,0,0])
				if (predictions[0,0,2]+predictions[0,0,3]>0.55) :
					ori= ori*(1+real_trend)
				elif predictions[0,0,1]+predictions[0,0,0]>0.55 :
					ori= ori*(1-real_trend)
		all_predictions = all_predictions[1:]
		all_output = all_output[1:]
		#print all_output.shape, all_predictions.shape
		expand = np.expand_dims( np.argmax( all_predictions, axis=2), axis=1)
		#print expand.shape
		acc= np.mean( 1*np.equal(expand ,all_output) )
		
		print 'test acc:%f, look_back earning:%f' % (acc, ori)	

def get_testdata(config):
	test_day = config.testroot+('stock_market_data_day-%s.csv'%config.ticker)
	test_week= config.testroot+('stock_market_data_week-%s.csv'%config.ticker)
	config.df_day = pd.read_csv(test_day)
	config.df_week = pd.read_csv(test_week)
	config.total_day_1y = total_day_1y = 52*5- config.week_unroll*5
	config.total_week_1y = total_week_1y=52 - config.week_unroll
	
	pre_test_data_day  =  config.df_day.loc[:,['Mid','Volume']].as_matrix()
	pre_test_data_week =  config.df_week.loc[:,['Mid','Volume']].as_matrix()

	#align pre_train_data_data with data week
	config.align=0
	while(config.df_day['Date'].loc[config.align] <=config.df_week['Date'].loc[0]):
	    config.align=config.align+1
	pre_test_data_day = pre_test_data_day[config.align:]

	config.test_data_day  = np.zeros(( pre_test_data_day.shape[0]- total_day_1y ,2))
	config.day_week_map    = np.zeros(( pre_test_data_day.shape[0]-total_day_1y )) #map day to corresponing week data
	config.test_data_week = np.zeros(( pre_test_data_week.shape[0]-total_week_1y,2))
	
	average_day  = np.mean( pre_test_data_day[:total_day_1y], axis=0) 
	average_week = np.mean( pre_test_data_week[:total_week_1y], axis=0) 
	average_fiveday = average_day
	
	config.five_day_avg      = np.copy(pre_test_data_day)
	for i in range(5, config.five_day_avg.shape[0]):
		config.five_day_avg[i]= config.five_day_avg[i-1]+0.2*(pre_test_data_day[i]-pre_test_data_day[i-5])
		
	config.five_day_avg_norm = np.zeros(( config.five_day_avg.shape[0]-total_day_1y ,2))
	config.ori_data_day = np.copy( pre_test_data_day[total_day_1y:])
	
	config.ori_data_day[0] = (pre_test_data_day[total_day_1y]+pre_test_data_day[total_day_1y-1]+ pre_test_data_day[total_day_1y-2])/3
	for i in range( 1, config.ori_data_day.shape[0]):
		config.ori_data_day[i]= (pre_test_data_day[i+total_day_1y]-pre_test_data_day[i-config.n_day_smooth+total_day_1y])*(1.0/config.n_day_smooth)+ config.ori_data_day[i-1]
	j=0
	for i in range(0, config.test_data_day.shape[0]): 
		if j>= config.test_data_week.shape[0]:
			config.day_week_map[i]=j
			#print i, j
		else:
			while ( config.df_day['Date'].loc[i+config.align] >config.df_week['Date'].loc[j]):
				j= j+1
				if j>= config.test_data_week.shape[0]:
					config.day_week_map[i] =j 
					break
				
			if j<config.test_data_week.shape[0]:	
				config.day_week_map[i] =j-1
				#print i, j-1

	EMA = np.array(0)
	for i in range(1, config.test_data_day.shape[0]):
		average_day = average_day+ (pre_test_data_day[i+total_day_1y]- pre_test_data_day[i])/ total_day_1y
		config.test_data_day[i,0] =  (config.ori_data_day[i,0]- config.ori_data_day[i-1,0])/(config.ori_data_day[i-1,0])
		config.test_data_day[i,1:] = (config.ori_data_day[i,1:]/(average_day[1:]))#*0.95+ 0.05*EMA
		EMA = config.test_data_day[i, 1:]
	EMA = np.array(0)
	for i in range(0, config.test_data_week.shape[0]):
		average_week = average_week+ (pre_test_data_week[i+total_week_1y]- pre_test_data_week[i])/ total_week_1y
		config.test_data_week[i,0] =  (pre_test_data_week[i+total_week_1y,0]- pre_test_data_week[i-1+total_week_1y,0])/(pre_test_data_week[i-1+total_week_1y,0])
		config.test_data_week[i,1:] = (pre_test_data_week[i+total_week_1y,1:]/(average_week[1:]))#*0.95+ 0.05*EMA
		EMA = config.test_data_week[i, 1:]
		
	EMA = np.array(0)
	for i in range(0,config.five_day_avg_norm.shape[0]):
		average_fiveday = average_fiveday+ (config.five_day_avg[(i+total_day_1y)]- config.five_day_avg[i])/ total_day_1y
		config.five_day_avg_norm[i,0]  =   (config.five_day_avg[i+total_day_1y,0]- config.five_day_avg[i+total_day_1y-1*5,0])/(config.five_day_avg[i+total_day_1y-1*5,0])
		config.five_day_avg_norm[i,1:] =   (config.five_day_avg[i+total_day_1y,1:]/(average_fiveday[1:]))#*0.95+ 0.05*EMA
		EMA = config.five_day_avg_norm[i, 1:]
	config.macd= np.zeros((config.test_data_day.shape[0], config.test_data_day.shape[1]))
	
	config.test_data_day= config.test_data_day/ config.train_data_max_day
	#for i in range(config.test_data_day.shape[0]):
		#print '%s\t%f\t%f\t%f' %( config.df_day['Date'].loc[i+config.align+total_day_1y],config.test_data_day[i,0],  config.test_data_day[i,1],config.ori_data_day[i,0])
	config.test_data_week= config.test_data_week/ config.train_data_max_week
	config.five_day_avg_norm= config.five_day_avg_norm/ config.train_data_max_day
	short= 0
	long_= 0
	for i in range(config.macd.shape[0]):
		short = pre_test_data_day[i+total_day_1y]*0.15+ short*0.85
		long_  = pre_test_data_day[i+total_day_1y]*0.075+ long_*0.925
		config.macd[i] = (short-long_)/pre_test_data_day[i+total_day_1y]
	config.test_data_day = np.concatenate( [config.test_data_day, config.macd], axis=1)
	
	#print config.test_data_week.shape
	if ( config.test_data_day.shape[0]>=config.day_unroll+1 and config.test_data_week.shape[0]>=config.week_unroll+1 and  config.day_week_map[-1]>=config.week_unroll): return True
	else: return False
	
def test_onestep(config, index):
	one_test_day,one_test_week = [],[]
        for i in range(config.day_unroll, 0, -1):
			one_test_day.append(config.test_data_day[index-i+1]) 
        for i in range(config.week_unroll,0, -1):
			if not config.output_normalize_dict[0,0]==1:
				week_data = config.test_data_week[int(config.day_week_map[index]-i)]
			else:
				week_data = config.five_day_avg_norm[int(index)-(i-1)*5]
			one_test_week.append(week_data)
			 
	result =  (config.ori_data_day[int(index+config.output_normalize_dict[0,0]),0]-config.ori_data_day[index,0])/config.ori_data_day[index,0]

	batch_labels  = np.zeros(   (1,int(config.output_normalize_dict.shape[0]),1) , dtype=np.float32)
	if config.predict_type=='binary': batch_labels[0,0,0]=(1 if result>=0 else 0)   
	if config.predict_type=='4class':
		if  (result>config.output_normalize_dict[0,1]):     
			batch_labels[0,0,0]=3
		elif(result>0):                               
			batch_labels[0,0,0]=2
		elif(result<-1*config.output_normalize_dict[0,1]):  
			batch_labels[0,0,0]=0
		else:  
			batch_labels[0,0,0]=1
	
	one_test_day  =  np.expand_dims( np.array(one_test_day),axis=0)
	one_test_week =  np.expand_dims( np.array(one_test_week),axis=0)
	
	return one_test_day, one_test_week, batch_labels

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--api_key',    type=str, default='J0LLMPESJ7H0SKXQ')
	parser.add_argument('--ticker',      type=str, default='SPX', help='the symbol of the stock')
	parser.add_argument('--iterations'   , type=int, default=2000)
	parser.add_argument('--test_sz' ,    type=int, default=533, help='the number of tests during training')
	parser.add_argument('--bz_size' ,     type=int, default=128, help='the size of batch of each training step')
	parser.add_argument('--n_day_smooth', type=int, default=3, help='to smooth the curve of daily price')       
	parser.add_argument('--day_unroll',   type=int, default=20, help='the number of previous days to predict the movement of the stock')
	parser.add_argument('--week_unroll',  type=int, default=12, help='the number of previous weeks to predict the movement of the stock')
	parser.add_argument('--predict_type', type=str, default='binary', choices=['binary', '4class'])	
	parser.add_argument('--day',          type=int, default=1, choices=[1, 5, 20], ) #1day, 1week, 1m	
	parser.add_argument('--mode',         type=str, default='train', choices=['train', 'test'])
	parser.add_argument('--ch_pt',        type=str, default='checkpoints/', help='the path of trained model for testing')
	parser.add_argument('--testroot',     type=str, default='test/', help='the path of test data for testing')
	parser.add_argument('--train_data_max_day', default=[0.0081, 1.619])
	parser.add_argument('--train_data_max_week', default=[0.0085, 1.46])
	                           
	config = parser.parse_args()
	config.train_data_max_day = np.array(config.train_data_max_day)
	config.train_data_max_week = np.array(config.train_data_max_week)	
	#day                #1week              #4week
	if config.day ==1:
		config.output_normalize_dict = np.array([ [1,0.003,0.0085]])
	elif config.day==5:
		config.output_normalize_dict = np.array([ [5,0.015,0.00191]])
	elif config.day==20:
		config.output_normalize_dict = np.array([ [20,0.04,0.038]])
                                   
	############# LSTM model architecture
	#lstm with 3 layers each with dimension as followed
	config.num_nodes = [300,200,250]
	config.n_layers = len(config.num_nodes) # number of layers
	config.dropout = 0.2 # dropout amount
	#########learning paramters
	config.learning_rate= 0.01
	config.min_learning_rate= 0.001
	
	

	if config.mode == 'train':
		download_preprocess_data(config) # download and preprocess data
		dg = DataGeneratorSeq(config.train_data_day, config.train_data_week, config.bz_size , \
                 config.day_unroll, config.week_unroll,config.output_normalize_dict, config.day_week_map, config.test_sz,  config.ori_data_day, config.ori_data_week,\
                  config.five_day_avg,config.five_day_avg_norm,predict_type=config.predict_type)
		train(config, dg)
	if config.mode =='test':
		test(config)
		
				
if __name__ == '__main__':
		main()


