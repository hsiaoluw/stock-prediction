Stock Movement Predictions with price and volume
======================================
This project can
* Predict the movement of the target stock with price and volume 
* Support the predicitons of 1 day, 1 week(5T), 1 month (20T), usage --day={1,5,20}
* Predict the movement with --predict_type={binary,4class}, there are 2 types support,  binary, 4class
	1. binary : tries to predict the movement with 2 kinds, up, down
	2. 4class： tries to predict the movement with 4 kinds, up, strong up, down, strong down

You can also tune other parameters, use 
```
python2 stock_pred.py --help 
```
for more infomation. stock_pred.py is in the folder clean python.

How to implement?
====================================
You can run stock_all_pred.ipynb with jupyter notebook or
run with python files in clean_pyhton2. Use the following command.

```
python2 stock_pred.py --predict_type=<type> --day=<day>
```
Replace \<type\> with either 4class, binary, and \<day\> with 1,5, or 20

For testing, please place the data in clean_python2/test/ folder.
Include two files, one with the day information, the other with the week information.
The file format should be 
```
Date       |Mid  |Volume 
2017-4-24  |2373 |369000000
2017-4-25  |2386 |399520000
......
```
You can see the example files in clean_python2/test/ for more detail. 
The number of daily data samples should be at least more than one year (5*52 days), and the number of weekly data samples
should be at leat more than one year (52 weekys).

The tesing command is as followed.
```
python2 stock_pred.py --mode=test --day=<# days of predictions> --ch_pt=checkpoints/model_binary_1_.ckpt
```
'--chpt=' is the chekcing point path you stored after training.

If the predict_type is 'binary',the reuslt would be <br> \<date down up \>.
If the predict_type is '4class',the result would be <br> \<date strong_down down up strong_up\>.

Results:
======================
Since we only care about the movement, we don't output the continuous price value prediction.
Instead, we output with discrete classes. The best way to measure our results
is the accuracy of predictions comparing to the actual movement. 
For 4class, we also provied the accuracy of right direction predictions called test_acc_s.
Both up, strong up are classified as positive, and down, strong down are negative.

A sample plot is below. The test_acc is larger than train_acc since we have dropout mechanism for training.
![accuracy_sample](accuracy_sample.png)
The final test accuracy after 1000 iterations:
```
       		|1d  | 5d | 20d
binary 		|0.87|0.75|0.73
4class 		|0.72|0.58|0.51
4class_s        |0.83|0.64|0.67
```

Other Details:
====================================
We have preprocessed the input for our training.
Instead of using the price value directly, we use percentange of changing comparing to previous day.
Normalize the volume, and price with the maxmum value.
The plot shows the price, volume after normalizaiton.
![Normalized data](normalized_pv.png)
We not only use the day sequence data, but also week sequence data together, as a result, we need two LSTMs.
The sequence of day by default is 20, while the sequence of week by default is 12. One can change the value by flag
--day_unroll=<# days>, --week_unroll=<# weeks>

 



