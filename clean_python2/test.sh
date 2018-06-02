python2 stock_pred.py --mode=test --day=1 --ch_pt=checkpoints/model_binary_1_.ckpt>predict_binary_1.txt
python2 stock_pred.py --mode=test --day=5 --ch_pt=checkpoints/model_binary_5_.ckpt>predict_binary_5.txt
python2 stock_pred.py --mode=test --day=20 --ch_pt=checkpoints/model_binary_20_.ckpt>predict_binary_20.txt
python2 stock_pred.py --mode=test --day=1 --predict_type=4class --ch_pt=checkpoints/model_4class_1_.ckpt>predict_4class_1.txt
python2 stock_pred.py --mode=test --day=5 --predict_type=4class --ch_pt=checkpoints/model_4class_5_.ckpt>predict_4class_5.txt
python2 stock_pred.py --mode=test --day=20 --predict_type=4class --ch_pt=checkpoints/model_4class_20_.ckpt>predict_4class_20.txt
