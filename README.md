[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BeBerasategi/symbol_demo/HEAD?urlpath=voila%2Frender%2Fapp%2FDemo.ipynb)

# Estudio, desarrollo y evaluación de técnicas de aprendizaje automático para el reconocimiento de símbolos matemáticos escritos a mano

Este repositorio complementa mi Trabajo de Fin de Grado, que ha consistido en entrenar varios **modelos de aprendizaje automático** para que sean capaces de clasificar imágenes de **símbolos matemáticos escritos a mano**. Se han utilizado imágenes de la base de datos [HASY](https://arxiv.org/pdf/1701.08380), que cuenta con símbolos de 369 clases distintas. Tras entrenar y evaluar los modelos, se ha construido una demo que permite interactuar con el sistema que mejor rendimiento ha tenido.

<p align="middle">
  <img src="Databases/HASY/hasy-overview_1.png" width="800" />
</p>

## Uso

El directorio `Databases` contiene las bases de datos con las que se han entrenado los modelos. En el directorio `HASY` se encuentran los archivos relativos a la base de datos original. `hasy_tools.py` contiene las funciones escritas por el autor de la base de datos HASY, que se han actualizado y modificado según los objetivos de este proyecto. Las versiones actualizadas están en el archivo `hasy_tools_updated.py`. En la segunda parte del trabajo ha sido necesario utilizar métodos de aumento de datos para obtener una base de datos en la que todas las clases tienen el mismo número de muestras. Todo lo relacionado con este proceso se puede encontrar en `Data Augmentation`.

Los programas utilizados para crear los modelos se encuentran dentro del directorio `models`:
1. Máquinas de vectores soporte (SVM).
2. Random forests.
3. Perceptrones multi capa.
4. Redes neuronales convolucionales.
Para cada modelo primero se ha realizado un ajuste de hiperparámetros (*grid search*), seguida de una validación cruzada (*cross validation*).

El directorio `models` contiene también los notebooks de Jupyter `Imagenes_entrenamiento.ipynb` y `Precisión_sensibilidad.ipynb` con los que se han generado los datos e imágenes utilizados en la sección de la memoria escrita que explica los resultados obtenidos.

Finalmente, se ha construido una demo que permite probar el sistema que mejor ha funcionado, una red neuronal convolucional entrenada en la base de datos aumentada. Lo necesario para hacer funcionar esta aplicación está en el directorio `app`. La demo se ejecuta abriendo el notebook `Demo.ipynb` utilizando `voila`. El enlace de `Binder` que se puede encontrar al principio de este documento facilita ese proceso.

## Paquetes necesarios

En este trabajo se ha utilizado `Python 3.8.10`. La lista de los paquetes que han sido necesarios se muestra a continuación:
```
Paquete                       Versión
----------------------------- --------------
matplotlib                    3.6.3
numpy                         1.23.5
pandas                        1.5.3
scikit-learn                  1.2.0
tensorflow                    2.12.0
keras                         2.11.0
ipywidgets                    8.0.4
ipycanvas                     0.13.2
voila                         0.5.7
```

## Licencia

`CC BY 4.0`
