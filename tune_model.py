import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from sklearn import metrics
from my_utils import load_data, my_plot_ana
from keras_tuner import HyperModel
import time
import keras_tuner as kt
import argparse
import shap
import notifyemail as notify

# set the email system // use your own information
notify.Reboost(mail_host=' ', mail_user=' ', mail_pass=' ',
               default_reciving_list=[' '], log_root_path='log', max_log_cnt=5)
notify.add_text("tuning results")
notify.send_log()

'''
tuning cnn and gru model.
using soft_max so that LRP and SHAP method both can use the 
best tuning models.
'''
parser = argparse.ArgumentParser(description='Run tuning experiment.')
parser.add_argument('-m', '--model', type=str, help='model name: cnn, gru')
parser.add_argument('-e', '--epochs', type=int, default=150, help='int, number of epoch')
parser.add_argument('-t', '--max_trials', type=int, default=15, help='int, number of max trials of tuning')


class CNN1DHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build_model(self, hp):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv1D(filters=hp.Int(f"units0", min_value=2, max_value=768, step=2),
                                    kernel_size=hp.Int(f"kernel0", min_value=1, max_value=15, step=2),
                                    padding='same', activation="relu", input_shape=self.input_shape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling1D())
        model.add(keras.layers.Dropout(rate=hp.Float(f"dropout_rate0", min_value=0, max_value=0.9, step=0.1)))

        i = hp.Int(f"layer_num", min_value=2, max_value=3, step=1)
        for i in range(1, i):
            model.add(keras.layers.Conv1D(filters=hp.Int(f"units_{i}", min_value=2, max_value=768, step=2),
                                    kernel_size=hp.Int(f"kernel_{i}", min_value=1, max_value=15, step=2),
                                    padding='same', activation="relu"))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.MaxPooling1D())
            model.add(keras.layers.Dropout(rate=hp.Float(f"dropout_rate_{i}", min_value=0, max_value=0.9, step=0.1)))

        model.add(keras.layers.Dense(hp.Int("dense_unit_0", min_value=2, max_value=768, step=2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(rate=hp.Float(f"dropout_rate_last", min_value=0, max_value=0.9, step=0.1)))
        model.add(keras.layers.Dense(2, activation='softmax'))

        learning_rate = hp.Float("lr", min_value=1e-6, max_value=1e-2, sampling="log")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


class GRUHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build_model(self, hp):
        model = keras.models.Sequential()
        model.add(keras.layers.GRU(hp.Int(f"units_0", min_value=2, max_value=768, step=2),
                                   return_sequences=True, activation="relu", input_shape=self.input_shape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling1D())
        model.add(keras.layers.Dropout(rate=hp.Float(f"dropout_rate0", min_value=0, max_value=0.9, step=0.1)))

        i = hp.Int(f"layer_num", min_value=1, max_value=3, step=1)
        for i in range(1, i):
            model.add(keras.layers.GRU(hp.Int(f"units_{i}", min_value=2, max_value=768, step=2),
                                       return_sequences=True, activation="relu"))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.MaxPooling1D())
            model.add(keras.layers.Dropout(rate=hp.Float(f"dropout_rate_{i}", min_value=0, max_value=0.9, step=0.1)))

        model.add(keras.layers.Dense(hp.Int("dense_unit_last", min_value=2, max_value=768, step=2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(rate=hp.Float(f"dropout_rate_last", min_value=0, max_value=0.9, step=0.1)))

        # softmax output
        model.add(keras.layers.Dense(2, activation="softmax"))

        learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


if __name__ == '__main__':
    args = parser.parse_args()
    model_name = args.model
    epochs = args.epochs
    max_trials = args.max_trials

    data_path = {
        'cnn': './data/o_stride_data_nor.mat',
        'gru': './data/e_stride_data_nor.mat'
    }
    # load data
    path = data_path[model_name]
    X_train, X_test, X_val, y_train, y_test, y_val = load_data(path, 0.2, 0.2)

    input_shape = X_train.shape[1:]

    # hyper model
    classifiers = {
        'cnn': CNN1DHyperModel(input_shape=input_shape),
        'gru': GRUHyperModel(input_shape=input_shape),
    }
    hp_model = classifiers[model_name]

    # tune model
    objective_metric = 'val_accuracy'
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)  # for multi gpu in one machine
    # strategy = tf.distribute.MultiWorkerMirroredStrategy() # for high performance cluster
    # # model tuning
    tuner = kt.BayesianOptimization(
        hypermodel=hp_model.build_model,
        objective=objective_metric,
        max_trials=max_trials,
        executions_per_trial=1,
        overwrite=True,
        project_name=model_name,
        distribution_strategy=strategy,
        directory=f'./tuning_log/{model_name}_{int(round(time.time()*1000))}'
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor=objective_metric, patience=35)

    tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val),
                 callbacks=[stop_early], verbose=2, use_multiprocessing=True)

    best_model = tuner.get_best_models(1)[0]

    best_hps = tuner.get_best_hyperparameters(1)[0]
    print(best_model.summary())
    print(f'best parameters is {best_hps.values}')
    # save model
    current_time = time.strftime("%m_%d_%H_%M")
    best_model.save(f'./result/models/{model_name}_{current_time}.h5')

