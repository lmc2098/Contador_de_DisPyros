#!/usr/bin/env python3
"""Entrena con recortes WAV y busca posibles disparos en otro WAV."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks, resample_poly
from sklearn.ensemble import RandomForestClassifier
from pyAudioAnalysis import ShortTermFeatures

FS = 16000
FRAME = 160  # 10 ms
BEFORE = int(0.08 * FS)
AFTER = int(0.22 * FS)
N_ESTIMATORS = 200


def leer_wav(ruta):
    """
    Lee un archivo de audio WAV, normaliza su amplitud a flotantes [-1.0, 1.0],
    lo convierte a 'mono' si es 'estéreo' y lo remuestrea a la frecuencia fija de 16000 Hz.

    Args:
        ruta (str o Path): Ubicación del archivo WAV a leer.

    Returns:
        np.ndarray: Vector unidimensional de tipo float64 con las muestras del audio.

    Raises:
        ValueError: Si el archivo está vacío, no es finito, tiene un formato 
                    de datos incompatible o no se puede procesar.
    """
    fs, audio = wavfile.read(ruta)
    if audio.size == 0:
        raise ValueError(f"WAV vacío: {ruta}")
    if np.issubdtype(audio.dtype, np.unsignedinteger):
        # Si el audio está guardado como 'unsigned integer' (sin signo) 
        # calcula el punto medio del rango de valores posibles normaliza a [-1.0, 1.0]   
        punto_medio = (np.iinfo(audio.dtype).max + 1) / 2
        audio = (audio.astype(np.float64) - punto_medio) / punto_medio
    elif np.issubdtype(audio.dtype, np.signedinteger):   
        # Si el audio está guardado como 'signed integer' (con signo) convierte 
        # a 'float64' y normaliza a [-1.0, 1.0]                  
        audio = audio.astype(np.float64) / (np.iinfo(audio.dtype).max + 1) 
    elif np.issubdtype(audio.dtype, np.floating): 
        # Si el audio ya está guardado como tipo 'float' convierte a float64 para consistencia
        audio = audio.astype(np.float64)          
    else:
        raise ValueError(f"Formato WAV no compatible: {ruta}")
    if audio.ndim == 2:             
        # Si detecta dos canales (estéreo), calcula el promedio de ambos para 
        # fusionarlo en un único canal monocromático.
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError(f"Audio inválido: {ruta}")
    if fs != FS:
        # Si la frecuencia de muestreo original no es igual a la frecuencia objetivo (FS), 
        # calcula el Máximo Común Divisor y aplica un filtrado y rediseño polifásico (resample_poly) 
        # para cambiar el muestreo sin deformar el tono del audio.
        divisor = math.gcd(fs, FS)
        audio = resample_poly(audio, FS // divisor, fs // divisor)
    return audio


def energias(audio):
    """
    Divide la señal de audio en ventanas de tamaño FRAME y calcula
    la energía RMS (Root Mean Square) para cada una de ellas.

    Args:
        audio (np.ndarray): Vector de muestras de audio normalizado.

    Returns:
        np.ndarray: Vector con los niveles de energía calculados por cuadro.
    """
    relleno = (-len(audio)) % FRAME
    cuadros = np.pad(audio, (0, relleno)).reshape(-1, FRAME) # Rellena con ceros el final del audio y lo reestructura en una matriz cuyas 
                                                             # filas son ventanas de FRAME muestras.
    return np.sqrt(np.mean(cuadros ** 2, axis=1))


def recorte(audio, muestra):
    """
    Extrae un segmento de audio centrado en una muestra específica, aplicando 
    márgenes fijos de tiempo (BEFORE y AFTER) y rellenando con ceros si el 
    recorte excede los límites físicos del archivo.

    Args:
        audio (np.ndarray): Vector completo de muestras de audio.
        muestra (int): Índice de la muestra central donde ocurre el evento.

    Returns:
        np.ndarray: Segmento de audio aislado con una longitud fija de muestras.
    """
    inicio = muestra - BEFORE
    fin = muestra + AFTER
    izquierda = max(0, -inicio)
    derecha = max(0, fin - len(audio))
    return np.pad(audio[max(0, inicio):min(len(audio), fin)],
                  (izquierda, derecha))


def caracteristicas(audio, muestra):
    """
    Aísla un segmento de interés y extrae sus vectores de características de corto 
    plazo (como MFCCs o espectrales), consolidándolos a través de su media y desviación estándar.

    Args:
        audio (np.ndarray): Vector de muestras de audio.
        muestra (int): Posición del evento acústico en el audio.

    Returns:
        np.ndarray: Vector numérico unidimensional limpio que actúa como huella acústica.

    Raises:
        ValueError: Si los descriptores extraídos contienen valores indeterminados.
    """
    segmento = recorte(audio, muestra)
    # Función de la librería pyAudioAnalysis que extrae características del segmento de audio en 
    # ventanas de tiempo corto. En este caso, ventanas de 25 ms con un paso de 10 ms.
    f, _ = ShortTermFeatures.feature_extraction(
                                segmento, FS, int(0.025 * FS), int(0.010 * FS), deltas=False
                                )
    # Calcula la media y la desviación estándar de cada característica a lo largo de las ventanas.
    vector = np.concatenate((f.mean(axis=1), f.std(axis=1)))
    if not np.isfinite(vector).all():
        raise ValueError("Se obtuvieron características no finitas.")
    return vector


def entrenar(carpeta):
    """
    Escanea las subcarpetas de entrenamiento, extrae la huella acústica en el punto 
    de máxima energía de cada archivo WAV y entrena un modelo RandomForestClassifier.

    Args:
        carpeta (Path): Ruta base que contiene los directorios "disparo" y "no_disparo".

    Returns:
        RandomForestClassifier: El modelo de Machine Learning entrenado y listo para clasificar.

    Raises:
        ValueError: Si no se encuentran archivos o si algún audio es completamente silencioso.
    """
    xs, ys = [], [] # Listas de características y etiquetas ("no_disparo" = 0, "disparo" = 1)
    for clase, etiqueta in (("no_disparo", 0), ("disparo", 1)):
        archivos = sorted((carpeta / clase).rglob("*.wav"))
        if not archivos:
            raise ValueError(f"No se encontraron WAV en {carpeta / clase}")
        for ruta in archivos:
            audio = leer_wav(ruta)
            energia = energias(audio)
            if energia.max() == 0:
                raise ValueError(f"Audio silencioso: {ruta}")
            # Encuentra el bloque con mayor volumen (np.argmax), lo mapea de regreso a su 
            # posición original (en muestras de audio) e intenta posicionarse exactamente 
            # en la mitad de ese bloque para asegurar que captura el estallido del sonido.
            muestra = min(int(np.argmax(energia)) * FRAME + FRAME // 2, len(audio) - 1)
            xs.append(caracteristicas(audio, muestra))
            ys.append(etiqueta)
        print(f"{clase}: {len(archivos)} recortes")
    # Se configura el clasificador creando 'n_estimators' árboles de decisión.
    # 'n_jobs=-1' utiliza todos los núcleos disponibles.
    modelo = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, class_weight="balanced", random_state=42, n_jobs=-1
    )
    modelo.fit(xs, ys)
    return modelo


def contar_disparos(modelo, audio, salida, umbral, prominencia, separacion):
    """
    Analiza un audio largo buscando picos de energía significativos mediante un umbral 
    de prominencia adaptativo, los clasifica usando el modelo entrenado y exporta 
    los resultados cronológicos a un archivo CSV.

    Args:
        modelo (RandomForestClassifier): Modelo entrenado para la detección.
        audio (np.ndarray): Vector de muestras de audio largo que se desea auditar.
        salida (Path): Ruta del archivo CSV donde se escribirá el reporte detallado.
        umbral (float): Probabilidad mínima (0.0 a 1.0) para validar un disparo positivo.
        prominencia (float): Porcentaje de la energía máxima requerido para detectar un pico de sonido.
        separacion (float): Tiempo de espera mínimo en segundos entre detecciones sucesivas.
    """
    energia = energias(audio)
    maximo = energia.max()
    if maximo == 0:
        indices = np.array([], dtype=int)
    else:
        # El relleno permite detectar picos en el primer y último cuadro.
        indices, _ = find_peaks(
            np.pad(energia, (1, 1)),
            prominence = prominencia * maximo,
            distance = max(1, math.ceil(separacion / 0.01)),
        )
        indices = indices - 1
        indices = indices[(indices >= 0) & (indices < len(energia))]

    total = 0
    with salida.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["tiempo_segundos", "puntaje", "clasificacion"])
        for indice in indices:
            muestra = min(int(indice) * FRAME + FRAME // 2, len(audio) - 1)
            x = caracteristicas(audio, muestra)
            puntaje = modelo.predict_proba([x])[0, 1] # Le solicita al Random Forest la probabilidad exacta (de 0.0 a 1.0) 
                                                      # de que este vector específico pertenezca a la clase positiva ("disparo" = 1)
            disparo = puntaje >= umbral
            total += int(disparo)
            # La siguiente línea inserta la primera fila de encabezados descriptivos 
            # en las columnas del reporte (tiempo_segundos, puntaje, clasificacion).
            escritor.writerow([f"{muestra / FS:.3f}", f"{puntaje:.4f}",
                                "posible_disparo" if disparo else "otro"])
    print(f"Candidatos: {len(indices)}")
    print(f"Posibles disparos: {total}")
    print(f"Detalle: {salida}")