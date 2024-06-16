# python3 SVM_grid_search.py > log_gs.txt 2> err_gs.txt & 
# DURACIÓN COMPLETA: 

# Se cambia el parámetro C para ver el efecto de la regularización.

import sys, os, time, functools
sys.path.append('/home/jovyan/work/databases/HASY/')
os.makedirs("./models_gs/", exist_ok=True)

from hasy_tools_updated import *

import matplotlib.pyplot as matplot
import numpy as np
import seaborn as sb
import pandas as pd

import sklearn.metrics as metrics
from sklearn.svm import LinearSVC

# Save the model:
from joblib import dump, load

start_time = time.time()
print = functools.partial(print, flush=True)

np.random.seed(42)
print('DESCRIPTION: linear svm training, with grid search for C.')
print('Start time:', time.strftime("%H:%M:%S", time.gmtime(start_time)), end='\n\n')

X_train, y_train, symbol_id2index_train, index2symbol_id_train = load_database('/home/jovyan/work/databases/HASY/benat-data/cv/fold-1/train.csv')
X_test, y_test, symbol_id2index_test, index2symbol_id_test = load_database('/home/jovyan/work/databases/HASY/benat-data/cv/fold-1/test.csv')
print("Data loaded")

y_train_id = np.array([index2symbol_id_train[element] for element in y_train])
y_test_id = np.array([index2symbol_id_test[element] for element in y_test])
print("Converted y indexes to symbol IDs")

# Data scaling
X_train = np.array(X_train)/255.0
y_train = y_train_id.astype(int)
X_test = np.array(X_test)/255.0
y_test = y_test_id.astype(int)
print("Data scaled", end='\n\n')

print("Grid search started-----------------")

acc = []
acc_tr = []
coefficient = []
for c in [0.00001,0.0001,0.001,0.01,0.1,1]:
    print(f"c={c}")
    svm = LinearSVC(dual=False, C=c)
    svm.fit(X_train, y_train)
    
    # Save model
    dump(svm, f'./models_gs/SVM_HASY_linear_C_{c}.joblib') 
    
    coef = svm.coef_
    
    p_tr = svm.predict(X_train)
    a_tr = metrics.accuracy_score(y_train, p_tr)
    
    pred = svm.predict(X_test)
    a = metrics.accuracy_score(y_test, pred)
    
    coefficient.append(coef)
    acc_tr.append(a_tr)
    acc.append(a)

print("Grid search finished-----------------", end='\n\n')

end_time = time.time()
elapsed_time = end_time - start_time
print('End time:', time.strftime("%H:%M:%S", time.gmtime(end_time)), end='\n\n')
print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

c = [0.00001,0.0001,0.001,0.01,0.1,1]

matplot.subplots(figsize=(10, 5))
matplot.semilogx(c, acc,'-D' ,color='red' , label="Testing Accuracy")
matplot.semilogx(c, acc_tr,'-gD' , label="Training Accuracy")
#matplot.xticks(L,L)
matplot.grid(True)
matplot.xlabel("Cost Parameter C")
matplot.ylabel("Accuracy")
matplot.legend()
matplot.title('Accuracy versus the Cost Parameter C (log-scale)')
matplot.savefig("Accuracy_vs_C_gs.png", dpi=200)

print("Plot made and saved")

print("Saving data...")

# Assuming c, acc_tr, and acc are lists of the same length
data = {
    "c": c,
    "acc_tr": [ '%.4f' % elem for elem in acc_tr],
    "acc": [ '%.4f' % elem for elem in acc]
}

df = pd.DataFrame(data)

# To save the dataframe to a csv file
df.to_csv('data_file.csv', index=False)

print("Data saved", end="\n\n")

best_c = c[np.argmax(acc)]
print("Best value for C: ",  best_c)
print(f"Best model: ./models_gs/SVM_HASY_linear_C_{best_c}.joblib")


