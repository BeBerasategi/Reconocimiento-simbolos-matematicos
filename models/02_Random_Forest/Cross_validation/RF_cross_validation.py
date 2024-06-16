# python3 RF_cross_validation.py > log_cv.txt 2> err_cv.txt & 
# DURACIÓN COMPLETA: 

# Se utiliza la validación cruzada en 5 particiones para obtener unas estadísticas más fiables del rendimiento del modelo RF.

import sys, os, time, functools
sys.path.append('/home/jovyan/work/databases/HASY/')

from hasy_tools_updated import *

import numpy as np
import pandas as pd

import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier 

start_time = time.time()
print = functools.partial(print, flush=True)

np.random.seed(42)
print('DESCRIPTION: Random Forest training, with cross validation to obtain more robust statistics.')
print('Start time:', time.strftime("%H:%M:%S", time.gmtime(start_time)), end='\n\n')

# Save the test set accuracies:
test_acc = {1:[],3:[],5:[], 'MER':[]}

# 5 folds will be used:
for fold in [1,2,3,4,5]:
    X_train, y_train, symbol_id2index_train, index2symbol_id_train = load_database(f'/home/jovyan/work/databases/HASY/benat-data/cv/fold-{fold}/train.csv')
    X_test, y_test, symbol_id2index_test, index2symbol_id_test = load_database(f'/home/jovyan/work/databases/HASY/benat-data/cv/fold-{fold}/test.csv')
    print(f"Fold {fold} loaded")

    y_train_id = np.array([index2symbol_id_train[element] for element in y_train])
    y_test_id = np.array([index2symbol_id_test[element] for element in y_test])
    print("Converted y indexes to symbol IDs")

    # Data scaling is not done (not needed for Random Forest)
    X_train = np.array(X_train) #/255.0
    y_train = y_train_id.astype(int)
    X_test = np.array(X_test) #/255.0
    y_test = y_test_id.astype(int)
    # print("Data scaled", end='\n\n')

    # Use the best model obtained in GS:
    rf=RandomForestClassifier(n_estimators=100, max_depth=45)
    rf.fit(X_train,y_train_id)

    y_proba = rf.predict_proba(X_test)
    y_pred = rf.predict(X_test)

    test_acc[1].append(metrics.accuracy_score(y_test_id, y_pred))
    test_acc['MER'].append(MER(y_test_id, y_pred))
    
    test_acc[3].append(metrics.top_k_accuracy_score(y_test_id, y_proba, k=3))
    test_acc[5].append(metrics.top_k_accuracy_score(y_test_id, y_proba, k=5))
   
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
