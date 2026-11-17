# Contador de disPyros

## 1. Descripción

Contador de disparos basado en un sistema acústico de detección. Desarrollado en Python con la librería Librosa (ampliar en caso de poder desarrollar la red neuronal).

## 2. Objetivo científico

Desarrollar una herramienta capaz de complementar pericias de escenas con disparos de armas de fuego.

## 3. Instalación

Es requerimiento contar con las librerías Numpy, Matplotlib y Librosa.
Colocar versión de Python y requirements.txt??

## 4. Uso

Se utiliza como entrada el archivo de audio que se quiere analizar.

El formato del archivo puede ser cualquiera que sea compatible con ```soundfile```. Por ejemplo:
* WAV
* FLAC
* OGG
* AIFF
* MAT
* ALAC
* MP3
* RAW

Algunos archivos incompatibles son:
* AAC
* M4A
* MP4
* WMA
* AC3
* WebM

En el caso de utilizar archivos en formato MP3, debe asegurarse que la versión de la librería ```libsndfile``` es 1.1.0 o superior. 

## 5. Ejemplo de ejecución

## 6. Metodología

Flujo del programa:
1. Lectura del archivo de audio
2. Aplicación de transformada de Fourier de tiempo corto (*Short-Time Fourier Transform* en inglés) STFT: Interesa únicamente la amplitud del espectro.
3. Conversión de amplitud a dB: Mejora la precisión ante los cambios de intensidad de sonido.
4. Cálculo de la diferencia de intensidad entre datos consecutivos: Se toma la intensidad en un tiempo dado y se le resta la intensidad del tiempo anterior. Esto se repite para todo el intervalo temporal.Interesan los tiempos en los cuales aumenta la intensidad, ya que puede indicar el inicio de un disparo. Por lo tanto, se descartan aquellas diferencias que resultan negativas.
5. Cálculo de la *novelty curve*: Se toma el promedio del cambio de intensidad de todas las frecuencias para los mismos tiempos.
6. Detección de los máximos de la *novelty curve*: Tomar todos los máximos como disparos es incorrecto. Deben evitarse máximos relacionados a la intensidad promedio de los datos y a otros sonidos que no son producidos por los disparos. Librosa provee un algoritmo que filtra máximos, quedándose con aquellos que están suficientemente separados y son suficientemente altos respecto de valores cercanos a los mismos.


Paso 4 y 5 los realiza automáticamente ```librosa.onset.onset_strenght```.

Paso 6 lo realiza ```librosa.util.localmax``` para detectar máximos y ```librosa.onset.onset_detect``` para filtrar los disparos.

## 7. Validación

Figura de mérito. Contar manualmente disparos en Audacity y comparar con los obtenidos con Librosa.

## 8. Autores

* **Lucas Cabral** ([@lmc2098](https://github.com))
* **Justo Gonzalez Litardo** ([@jutogl](https://github.com))

## 9. Licencia

Este proyecto está bajo la licencia MIT. Consulte el archivo [LICENSE](LICENSE) para más detalles.

Copyright (c) [2026] [Lucas Cabral y Justo Gonzalez Litardo]