import tensorflow as tf
import datetime
from keras.layers import LSTM, Dense,Dropout
from sklearn.model_selection import train_test_split
import numpy as np,pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorboard.plugins.hparams import api as hp
from keras.models import Sequential
df = pd.read_csv('2015-2022bitcoinprice.csv',sep=';', parse_dates=['date'],index_col=0)
scaler=MinMaxScaler()   
df['ScaledPrice'] =scaler.fit_transform( df[['price']])                
df['Prediction']=df['ScaledPrice'].shift(-1)
df=df[:-1]
X = np.array(df['ScaledPrice'])
y = np.array(df['Prediction'])
X_train, x_test, y_train, y_test=  train_test_split(X, y, test_size=0.2,shuffle=False)
X_train = np.reshape(X_train, (X_train.shape[0], 1, 1))
x_test = np.reshape(x_test, (x_test.shape[0], 1, 1))
epochs=100

HP_OPTIMIZER=hp.HParam('optimizer', hp.Discrete(['adam', 'sgd','Adadelta', 'Adagrad','rmsprop','Adamax']))
HP_ACTIVATION=hp.HParam('activation', hp.Discrete(['tanh', 'softmax', 'relu', 'sigmoid']))
METRIC_MSE = 'mean_squared_error'
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#!rm -rf ./logs/ 
writer = tf.summary.create_file_writer(log_dir)

with writer.as_default():   
    hp.hparams_config(
        hparams=[HP_ACTIVATION, HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC_MSE, display_name='MSE')]
    )                  
def create_model(hparams):
    activation=hparams[HP_ACTIVATION]
    if activation=='tanh':
        activation=tf.nn.tanh
    elif activation=='softmax':
        activation=tf.nn.softmax
    elif activation=='relu':
        activation=tf.nn.relu
    elif activation=='sigmoid':
        activation=tf.nn.sigmoid
    else:
        raise ValueError("unexpected activation name: %r" % (activation)) 
    model=Sequential()
    model.add(LSTM(64,input_shape=(1, 1),go_backwards=False,activation=activation))  # returns a sequence of vectors of dimension 32
    model.add(Dropout(0.2)) 
    model.add(Dense(1))
    optimizer  = hparams[HP_OPTIMIZER]
    if optimizer == "adam":
        optimizer = tf.optimizers.Adam()
    elif optimizer == "sgd":
        optimizer = tf.optimizers.SGD()
    elif optimizer=='rmsprop':
        optimizer = tf.optimizers.RMSprop()
    elif optimizer == "Adadelta":
        optimizer = tf.optimizers.Adadelta()
    elif optimizer == "Adagrad":
        optimizer = tf.optimizers.Adagrad() 
    elif optimizer=='Adamax':
          optimizer = tf.optimizers.Adamax()    
    else:
        raise ValueError("unexpected optimizer name: %r" % (optimizer))
    model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['mse'])
    model.fit(X_train,y_train,batch_size=64,epochs=epochs,
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir),hp.KerasCallback(log_dir, hparams)])
    _,loss=model.evaluate(x_test, y_test)
    return loss
   
def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    METRIC_RMSE = create_model(hparams)
    #converting to tf scalar
    #loss= tf.reshape(tf.convert_to_tensor(METRIC_RMSE), []).numpy()
    tf.summary.scalar( 'mean_squared_error',METRIC_RMSE,step=2)
session_num = 0
for activation in HP_ACTIVATION.domain.values:
    for optimizer in HP_OPTIMIZER.domain.values:
        hparams = {
              HP_ACTIVATION: activation,
              HP_OPTIMIZER: optimizer,
        }
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run('logs/hparam_tuning/' + run_name, hparams)
        session_num += 1


