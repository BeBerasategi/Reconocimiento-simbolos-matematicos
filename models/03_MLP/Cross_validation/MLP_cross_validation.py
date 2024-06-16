# python3 MLP_cross_validation.py > log_cv.txt 2> err_cv.txt & 
# DURACIÓN COMPLETA: 

# Se utiliza la validación cruzada en 5 particiones para obtener unas estadísticas más fiables del rendimiento del modelo MLP.

import sys, os, time, functools
sys.path.append('/home/jovyan/work/databases/HASY/')
os.makedirs("./models_cv/", exist_ok=True)
os.makedirs("./plots_cv/", exist_ok=True)

from hasy_tools_updated import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier 

start_time = time.time()
print = functools.partial(print, flush=True)

print('DESCRIPTION: MLP training, with cross validation to obtain more robust statistics.')
print('Start time:', time.strftime("%H:%M:%S", time.gmtime(start_time)), end='\n\n')

# Save the test set accuracies:
test_acc = {1:[],3:[],5:[], 'MER':[]}

# 5 folds will be used:
for fold in [1,2,3,4,5]:
    X_train, y_train, symbol_id2index_train, index2symbol_id_train = load_database(f'/home/jovyan/work/databases/HASY/benat-data/cv/fold-{fold}/train.csv')
    X_test, y_test, symbol_id2index_test, index2symbol_id_test = load_database(f'/home/jovyan/work/databases/HASY/benat-data/cv/fold-{fold}/test.csv')
    print(f"Fold {fold} loaded----------------------------------------")

    # First, get the ID of each label:
    y_train_id = np.array([index2symbol_id_train[element] for element in y_train])
    y_test_id = np.array([index2symbol_id_test[element] for element in y_test])
    # Go back to indexes, but use the SAME indexing SYSTEM for all:
    y_train = np.array([symbol_id2index_train[element] for element in y_train_id])
    y_test = np.array([symbol_id2index_train[element] for element in y_test_id])

    # Data scaling
    X_test, X_train = X_test / 255., X_train / 255.
    y_test, y_train =  y_test.astype(int), y_train.astype(int)
    print("Data scaled", end='\n\n')

    # Reset keras session:
    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    # Use the best model obtained in GS:
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[32*32]),
        keras.layers.Dense(800, activation="relu"),
        keras.layers.Dense(800, activation="relu"),
        keras.layers.Dense(369, activation="softmax")
    ])
    lr = 0.02

    model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(learning_rate=lr),
              metrics=["accuracy"])
    # Callbacks
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,  monitor='loss', restore_best_weights=True)
    # This times, a validation set will not be needed.
    history = model.fit(X_train, y_train, epochs=30, # It should be enough with ~18 epochs.
                        #validation_data=(X_valid, y_valid),
                        callbacks=[early_stopping_cb],
                        verbose=2)
    
    print("The model has been trained.")

    #Save history
    df = pd.DataFrame(history.history)
    df.to_csv(f'./models_cv/history_fold_{fold}.csv')

    # Plot:
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 4)
    plt.savefig(f"./plots_cv/learning_curves_plot_fold_{fold}.png", dpi=200)

    # Test set accuracy: note that y_test are the symbol indexes, not the IDs.
    y_proba = model.predict(X_test, verbose=2)
    y_pred = np.argmax(y_proba, axis=-1)
    y_pred_id = np.array([index2symbol_id_train[element] for element in y_pred])

    test_acc[1].append(metrics.accuracy_score(y_test, y_pred))
    test_acc['MER'].append(MER(y_test_id, y_pred_id))
    
    test_acc[3].append(metrics.top_k_accuracy_score(y_test, y_proba, k=3))
    test_acc[5].append(metrics.top_k_accuracy_score(y_test, y_proba, k=5))
   
    print(f"Fold {fold} TOP1 accuracy: {test_acc[1][-1]}", end='\n')
    print(f"Fold {fold} MER accuracy: {test_acc['MER'][-1]}", end='\n\n')

print("Cross validation finished-----------------", end='\n\n')

end_time = time.time()
elapsed_time = end_time - start_time
print('End time:', time.strftime("%H:%M:%S", time.gmtime(end_time)), end='\n\n')
print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

# Assuming depth_list, acc_tr, and acc are lists of the same length
df = pd.DataFrame(test_acc)

# To save the dataframe to a csv file
df.to_csv('data_file_CV.csv', index=False)
df.describe().to_csv('data_file_CV_stats.csv')

print("Data saved", end="\n\n")

print("Top1 accuracy mean:", df[1].mean())
print("Top1 accuracy std:", df[1].std())
print("Top3 accuracy mean:", df[3].mean())
print("Top3 accuracy std:", df[3].std())
print("Top5 accuracy mean:", df[5].mean())
print("Top5 accuracy std:", df[5].std())
print("MER mean:", df['MER'].mean())
print("MER std:", df['MER'].std())

print("End of the script")

