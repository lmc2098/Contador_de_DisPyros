# Contador de disPyros

## 1. Descripción

Contadores de disparos basados en: 
- Un sistema acústico de detección. Desarrollado en Python con la librería _Librosa_ (ampliar en caso de poder desarrollar la red neuronal).
- Un modelo de _RandomForest_ previamente entrenado con características extraídas del audio de interés con funciones de la librería _pyAudioAnalysis_. 

## 2. Objetivo científico

Desarrollar una herramienta capaz de complementar pericias de escenas con disparos de armas de fuego.

## 3. Instalación

Este proyecto utiliza las siguientes librerías de Python:

* numpy
* matplotlib
* librosa
* pyAudioAnalysis
* plotly
+ librerías registradas en 'requirements.txt'

**Opción 1: usando uv**

1. Crear un entorno virtual:

        uv venv

2. Activar el entorno:

    **Linux/macOS**

        source .venv/bin/activate

    **Windows (PowerShell)**

        .venv\Scripts\Activate.ps1

3. Instalar las dependencias:

        uv pip install -r numpy matplotlib librosa pyAudioAnalysis plotly

**Opción 2: usando Conda**

1. Crear un entorno nuevo:

        conda create -n audio-env python=3.12

2. Activar el entorno:

        conda activate audio-env

3. Instalar las dependencias:

        conda install numpy matplotlib
        pip install librosa pyAudioAnalysis plotly

4. Verificación

    Comprobar que las librerías se instalaron correctamente:

        python -c "import numpy, matplotlib, librosa; print('Instalación correcta')"

## 4. Uso

Para realizar un conteo de disparos, debe ejecutarse el archivo ```contador_disparos.py```, dentro de la carpeta ```librosa```. La salida da la cantidad de disparos estimada por el programa y un gráfico que contiene:
* Señal del audio
* Picos de intensidad detectados
* Picos de intensidad clasificados como disparo

Se utiliza como entrada el archivo de audio que se quiere analizar. El archivo se carga modificando la variable ```archivo``` (dentro de ```contador_disparos.py```) por el nombre del archivo de audio que se desea analizar.

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

Para evaluar la eficiencia del contador de disparos, puede utilizarse ```validacion.py```. El archivo de audio se ingresa de la misma manera que en ```contador_disparos.py```, a la variable ```archivo```. Luego se necesita de un archivo de dos columnas (tiempo, intensidad) que contenga todos los disparos. Este útlimo archivo se carga en la variable ```etiquetas```. La salida devuelve los disparos detectados por el programa, los realmente efectuados y un gráfico que marca en qué instantes se detectaron disparos y en cuáles ocurrieron realmente.

## 5. Ejemplo de ejecución

La carpeta ```audios``` contiene un audio de la Masacre de Ramallo para probar el contador de disparos (```Audio 1 - Recorte Damian Kuc - Recortado.mp3```). Utilice ese archivo para ejecutar ```contador_disparos.py```. Modifique la variable ```archivo``` dentro de ```contador_disparos.py``` por  ```audio/Audio 1 - Recorte Damian Kuc - Recortado.mp3``` y ejecute el programa.

## 6. Metodología

Flujo del programa:
1. Lectura del archivo de audio
2. Aplicación de transformada de Fourier de tiempo corto (*Short-Time Fourier Transform* en inglés) STFT: Interesa únicamente la amplitud del espectro.
3. Conversión de amplitud a dB: Mejora la precisión ante los cambios de intensidad de sonido.
4. Cálculo de la diferencia de intensidad entre datos consecutivos: Se toma la intensidad en un tiempo dado y se le resta la intensidad del tiempo anterior. Esto se repite para todo el intervalo temporal. Interesan los tiempos en los cuales aumenta la intensidad, ya que puede indicar el inicio de un disparo. Por lo tanto, se descartan aquellas diferencias que resultan negativas.
5. Cálculo de la *novelty curve*: Se toma el promedio del cambio de intensidad de todas las frecuencias para los mismos tiempos.
6. Detección de los máximos de la *novelty curve*: Tomar todos los máximos como disparos es incorrecto. Deben evitarse máximos relacionados a la intensidad promedio de los datos y a otros sonidos que no son producidos por los disparos. Librosa provee un algoritmo que filtra máximos, quedándose con aquellos que están suficientemente separados y son suficientemente altos respecto de valores cercanos a los mismos.


Paso 4 y 5 los realiza automáticamente ```librosa.onset.onset_strenght```.

Paso 6 lo realiza ```librosa.util.localmax``` para detectar máximos y ```librosa.onset.onset_detect``` para filtrar los disparos.

## 7. Validación

La eficiencia del contador se analiza ejecutando ```validacion.py```. Se utilizan los archivos ```audio/Audio 1 - Recorte Damian Kuc - Recortado.mp3``` y ```\audios\Masacre de Ramallo - Test\Audio 1 - Etiquetas.txt```. La siguiente imagen contiene los resultados obtenidos.

PENDIENTE SUBIR IMAGEN

Para una secuencia de 105 disparos efectuados, se detectan 96. Esto no quiere decir que todos los sonidos clasificados como disparos sean efectivamente tal. El programa es capaz de detectar todos los disparos que están suficientemente separados temporalmente entre ellos y que no tienen algún sonido de interferencia. Se observa que el programa puede contabilizar interferencias (gritos, golpes, etc.) como disparos, además de no detectar disparos reales durante dicha interferencia. También se observa que cuando los disparos son suficientemente consecutivos, el programa no tiene la resolución temporal suficiente para detectarlos a todos.

## 8. Autores

* **Lucas Cabral** ([@lmc2098](https://github.com))
* **Justo Gonzalez Litardo** ([@justogl](https://github.com))

## 9. Licencia

Este proyecto está bajo la licencia MIT. Consulte el archivo [LICENSE](LICENSE) para más detalles.

Copyright (c) [2026] [Lucas Cabral y Justo Gonzalez Litardo]