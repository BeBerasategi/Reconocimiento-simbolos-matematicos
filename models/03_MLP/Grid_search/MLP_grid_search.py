# python3 MLP_grid_search.py > log_gs.txt 2> err_gs.txt & 
# DURACIÓN COMPLETA: 01:48:49

# Se cambian varios hiperparámetros para buscar la mejor combinación.

import sys, os, time, functools
sys.path.append('/home/jovyan/work/databases/HASY/')
os.makedirs("./models_gs/", exist_ok=True)
os.makedirs("./plots_gs/", exist_ok=True)

from hasy_tools_updated import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Save the model:
from joblib import dump, load

start_time = time.time()
print = functools.partial(print, flush=True)

np.random.seed(42)
tf.random.set_seed(42)
print('DESCRIPTION: MLP, with grid search for some hyperparameters.')
print('Start time:', time.strftime("%H:%M:%S", time.gmtime(start_time)), end='\n\n')

X_train, y_train, symbol_id2index_train, index2symbol_id_train = load_database('/home/jovyan/work/databases/HASY/benat-data/cv/fold-1/train.csv')
X_test, y_test, symbol_id2index_test, index2symbol_id_test = load_database('/home/jovyan/work/databases/HASY/benat-data/cv/fold-1/test.csv')
print("Data loaded")

# First, get the ID of each label:
y_train_id = np.array([index2symbol_id_train[element] for element in y_train])
y_test_id = np.array([index2symbol_id_test[element] for element in y_test])
# Go back to indexes, but use the SAME indexing SYSTEM for all:
y_train = np.array([symbol_id2index_train[element] for element in y_train_id])
y_test = np.array([symbol_id2index_train[element] for element in y_test_id])

# Data scaling
X_valid, X_train = X_test / 255., X_train / 255.
y_valid, y_train =  y_test.astype(int), y_train.astype(int)
print("Data scaled", end='\n\n')

print("Grid search started-----------------")

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model0 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[32*32]),
    keras.layers.Dense(800, activation="relu"),
    keras.layers.Dense(500, activation="relu"),
    keras.layers.Dense(369, activation="softmax")
])

model1 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[32*32]),
    keras.layers.Dense(500, activation="relu"),
    keras.layers.Dense(500, activation="relu"),
    keras.layers.Dense(369, activation="softmax")
])

model2 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[32*32]),
    keras.layers.Dense(800, activation="relu"),
    keras.layers.Dense(800, activation="relu"),
    keras.layers.Dense(369, activation="softmax")
])

model3 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[32*32]),
    keras.layers.Dense(500, activation="relu"),
    keras.layers.Dense(200, activation="relu"),
    keras.layers.Dense(369, activation="softmax")
])

lr = [0.04, 0.02, 0.01, 0.005]

dd
            'comb10' : (keras.models.clone_model(model1), lr[0]),
            'comb11' : (keras.models.clone_model(model1), lr[1]),
            'comb12' : (keras.models.clone_model(model1), lr[2]),
            'comb13' : (keras.models.clone_model(model1), lr[3]),
            'comb20' : (keras.models.clone_model(model2), lr[0]),
            'comb21' : (keras.models.clone_model(model2), lr[1]),
            'comb22' : (keras.models.clone_model(model2), lr[2]),
            'comb23' : (keras.models.clone_model(model2), lr[3]),
            'comb30' : (keras.models.clone_model(model3), lr[0]),
            'comb31' : (keras.models.clone_model(model3), lr[1]),
            'comb32' : (keras.models.clone_model(model3), lr[2]),
            'comb33' : (keras.models.clone_model(model3), lr[3])}


acc_tr = []
acc_val = []
for comb in CV_dict:
    print(f"Comb={comb}")
    model, lr = CV_dict[comb]
    model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(learning_rate=lr),
              metrics=["accuracy"])
    # Callbacks
    checkpoint_cb = keras.callbacks.ModelCheckpoint(f'./models_gs/MLP_HASY_{comb}.h5', save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=30,
                        validation_data=(X_valid, y_valid),
                        callbacks=[checkpoint_cb, early_stopping_cb],
                        verbose=2)
    
    # Best accuracy:
    index = np.argmax(history.history['val_accuracy'])
    acc_val.append(history.history['val_accuracy'][index])
    acc_tr.append(history.history['accuracy'][index])

    #Save history
    df = pd.DataFrame(history.history)
    df.to_csv(f'./models_gs/history_{comb}.csv')

    # Plot:
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 4)
    plt.savefig(f"./plots_gs/learning_curves_plot_{comb}.png", dpi=200)
    
    # Reset the system:
    del(model)
    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

print("Grid search finished-----------------", end='\n\n')

end_time = time.time()
elapsed_time = end_time - start_time
print('End time:', time.strftime("%H:%M:%S", time.gmtime(end_time)), end='\n\n')
print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

print("Saving data...")

# Assuming comb, acc_tr, and acc_val are lists of the same length
data = {
    "combination": list(CV_dict.keys()),
    "acc_tr": [ '%.4f' % elem for elem in acc_tr],
    "acc_val": [ '%.4f' % elem for elem in acc_val]
}

df = pd.DataFrame(data)

# To save the dataframe to a csv file
df.to_csv('data_file.csv', index=False)

print("Data saved", end="\n\n")

best_comb = list(CV_dict.keys())[np.argmax(data["acc_val"])]
model, lr = CV_dict[best_comb]
print("Best combination: ",  best_comb)
print(f"Best learning rate: {lr}")
print(f"Best model: {model}")
print(f"Best model: ./models_gs/MLP_HASY_{best_comb}.h5")

