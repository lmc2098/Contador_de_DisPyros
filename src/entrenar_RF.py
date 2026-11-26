#!/usr/bin/env python3
"""Entrena modelo de detección de disparos con recortes WAV"""

import argparse
import csv
import joblib
import json
import numpy as np
import math

from pathlib import Path
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from pyAudioAnalysis import ShortTermFeatures
from .utils import leer_wav, energias, recorte

FS = 16000
FRAME = 160  # 10 ms
BEFORE = int(0.08 * FS)
AFTER = int(0.22 * FS)
N_ESTIMATORS = 200

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
    segmento = recorte(audio, muestra, BEFORE, AFTER)
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
            energia = energias(audio, FRAME)
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

def main():
    parser = argparse.ArgumentParser(description="Entrena un modelo de detección de disparos.")
    parser.add_argument("--carpeta_entrenamiento", type=str, help="Ruta a la carpeta de entrenamiento")
    parser.add_argument("--carpeta_salida", type=str, help="Ruta para guardar el modelo entrenado")
    parser.add_argument("--nombre_modelo", type=str, default="modelo.joblib", help="Nombre del archivo del modelo entrenado")
    args = parser.parse_args()

    carpeta = Path(args.carpeta_entrenamiento)
    print(f"Entrenando clasificador RandomForest...")
    modelo = entrenar(carpeta)
    ruta_salida = Path(args.carpeta_salida) / (args.nombre_modelo + ".joblib")
    joblib.dump(modelo, ruta_salida)
    print(f"Modelo entrenado con éxito y guardado en {ruta_salida}.")
    parametros_RF = {
        "FS": FS,
        "FRAME": FRAME,
        "BEFORE": BEFORE,
        "AFTER": AFTER
        }
    with open(Path(args.carpeta_salida) / 'parametros_RF.json', 'w', encoding='utf-8') as f:
        json.dump(parametros_RF, f, indent=4)
    print(f"Parámetros de entrenamiento guardados en {Path(args.carpeta_salida) / 'parametros_RF.json'}")

if __name__ == "__main__":
    main()