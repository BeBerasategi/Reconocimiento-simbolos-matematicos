# python3 CNN_2layer_cross_validation.py > log_cv.txt 2> err_cv.txt & 
# DURACIÓN COMPLETA: 
# 23362

# Se utiliza la validación cruzada en 5 particiones para obtener unas estadísticas más fiables del rendimiento de un modelo en concreto de CNN.

# Modelo: 2-LAYER (artículo: "HASYv2 dataset").

import sys, os, time, functools
sys.path.append('/home/jovyan/work/databases/HASY/')
os.makedirs("./models_cv/", exist_ok=True)
os.makedirs("./plots_cv/", exist_ok=True)

from hasy_tools_updated import *

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import sklearn.metrics as metrics

start_time = time.time()
print = functools.partial(print, flush=True)

print('DESCRIPTION: CNN training, with cross validation to obtain more robust statistics. Model: 2-LAYER. Artículo: "HASYv2 dataset".')
print('Start time:', time.strftime("%H:%M:%S", time.gmtime(start_time)), end='\n\n')

# Save the test set accuracies:
test_acc = {1:[],3:[],5:[], 'MER':[], 'epochs':[]}

# 5 folds will be used:
for fold in [1,2,3,4,5]:
    X_train, y_train, symbol_id2index_train, index2symbol_id_train = load_database(f'/home/jovyan/work/databases/HASY/benat-data/cv/fold-{fold}/train.csv')
    X_test, y_test, symbol_id2index_test, index2symbol_id_test = load_database(f'/home/jovyan/work/databases/HASY/benat-data/cv/fold-{fold}/test.csv')
    print(f"Fold {fold} loaded----------------------------------------")

    # First, get the ID of each label:
    y_train_id = np.array([index2symbol_id_train[element] for element in y_train])
    y_test_id = np.array([index2symbol_id_test[element] for element in y_test])
    print("Converted y indexes to symbol IDs")
    # Go back to indexes, but use the SAME indexing SYSTEM for all:
    y_train = np.array([symbol_id2index_train[element] for element in y_train_id])
    y_test = np.array([symbol_id2index_train[element] for element in y_test_id])    
    print("Converted y symbol IDs to same index system")

    # Data scaling
    X_train, X_test = X_train / 255., X_test/ 255.
    y_train, y_test = y_train.astype(int), y_test.astype(int)
    print("Data scaled", end='\n\n')

    # Reshape the data to 2D:
    X_train = X_train.reshape(X_train.shape[0],32,32)[..., np.newaxis]
    X_test = X_test.reshape(X_test.shape[0],32,32)[..., np.newaxis]
    # Another option: tf.image.resize(images, [32,32])
    
    # Split the training data in train0 and validation0
    index_split = int(len(y_train)*70/80)
    X_train0 = X_train[:index_split]
    X_val0 = X_train[index_split:]
    y_train0 = y_train[:index_split]
    y_val0 = y_train[index_split:]

    # Reset keras session:
    keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Build the keras modeL: default Conv2 strides: (1,1)
    # MODEL: 2-LAYER
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2)), # Default values.
        keras.layers.Flatten(),
        keras.layers.Dense(369, activation="softmax")
    ])

    # Model summary:
    model.build(X_train.shape) 
    print(model.summary())
    
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])    

    # Callbacks
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,  monitor="val_accuracy", mode = "max", restore_best_weights=False) #IMPORTANT: Patience = 3.
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f"./models_cv/best_model_fold_{fold}.h5", monitor="val_accuracy", mode="max", save_weights_only = False, save_best_only=True, verbose=1)

    
    history = model.fit(X_train0, y_train0, epochs=20, callbacks=[early_stopping_cb, checkpoint],
                    verbose=2, 
                    validation_data=(X_val0, y_val0)) # Validation data is provided!

    print("The model has been trained.")
    
    
    #Save history
    df = pd.DataFrame(history.history)
    df.to_csv(f'./models_cv/history_fold_{fold}.csv')
    
    # Plot
    df.plot(figsize=(8, 5))
    plt.grid(True)
    #plt.gca().set_ylim(0, 1)
    plt.savefig(f"./plots_cv/learning_curves_plot_fold_{fold}.png", dpi=200)
    
    # Restore best weights:
    model.load_weights(f"./models_cv/best_model_fold_{fold}.h5")
    
    # Print best epoch number:
    n_epochs_best = np.argmax(history.history['val_accuracy'])+1 # Sumar uno, porque no empiezan de 0.
    test_acc['epochs'].append(n_epochs_best)
    print("Best epoch number: ", n_epochs_best)
    
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

print("Epoch mean:", df['epochs'].mean())
print("Epoch std:", df['epochs'].std())

print("End of the script")
