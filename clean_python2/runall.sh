python2 stock_pred.py --predict_type=binary --day=1 >binary_1day.txt
python2 stock_pred.py --predict_type=binary --day=5  --day_unroll=15 >binary_5day.txt
python2 stock_pred.py --predict_type=binary --day=20 --day_unroll=15 >binary_20day.txt
python2 stock_pred.py --predict_type=4class --day=1 >4class_1day.txt
python2 stock_pred.py --predict_type=4class --day=5  --day_unroll=15 >4class_5day.txt
python2 stock_pred.py --predict_type=4class --day=20 --day_unroll=15 >4class_20day.txt
