import argparse
import csv
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
import math
import librosa
import pandas as pd

from pathlib import Path
from scipy.signal import find_peaks
from .utils import energias
from .graphs import analisis_librosa
from .entrenar_RF import caracteristicas

def contar_disparos_librosa(audio, salida="conteo_disparos.csv", graficar=False, etiquetas_reales_csv=None):
    """
    Detecta disparos en un audio largo mediante el análisis de la curva de novedad de Librosa.

    Esta función procesa un archivo de audio calculando la Transformada de Fourier de Tiempo 
    Corto (STFT) y derivando la fuerza de inicio (onset strength) en decibelios como curva de 
    novedad espectral. Filtra los máximos locales para identificar transitorios acústicos abruptos, 
    exporta un reporte cronológico estandarizado en formato CSV y grafica los resultados usando la 
    funcion 'analisis_librosa' de 'graphs.py'.

    Args:
        audio (np.ndarray): Vector de muestras de audio largo que se desea auditar.
        salida (str o Path, opcional): Ruta del archivo CSV donde se guardará el reporte de 
            detecciones. Por defecto es "conteo_disparos.csv".
        graficar (bool, opcional): Indica si se debe renderizar el gráfico estático de subplots 
            en Matplotlib. Por defecto es False.
        etiquetas_reales_csv (str o Path, opcional): Ruta del archivo CSV que contiene las 
            anotaciones de laboratorio (disparos reales efectuados) para contrastar los 
            resultados. Por defecto es None.

    Returns:
        None: Escribe los resultados directamente en el disco e imprime métricas por consola.
    """
    # Carga de archivo
    audio, fs = librosa.load(audio, sr=None)

    # Short-Time Fourier Transform (STFT)
    stft = librosa.stft(audio)

    # Cambio de unidades a dB y calculo de curva de novedad
    logS = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    onset_env = librosa.onset.onset_strength(S=logS)

    # Filtrado de maximos de la curva de novedad para obtener disparos
    times = librosa.times_like(onset_env, sr=fs)
    onset_peaks = librosa.util.localmax(onset_env)
    onset_detect = librosa.onset.onset_detect(onset_envelope=onset_env)

    datos = np.empty((len(onset_detect), 3), dtype=object)
    datos[:, 0] = times[onset_detect]
    datos[:, 1] = '' # puntaje vacío para mantener la estructura, columna usada por el modelo RF
    datos[:, 2] = 'posible_disparo'

    if etiquetas_reales_csv is not None:
        etiquetas_reales = pd.read_csv(etiquetas_reales_csv)
        analisis_librosa(times, onset_env, onset_peaks, onset_detect, audio=audio, fs=fs, etiquetas_reales=etiquetas_reales["tiempo_segundos"], graficar=graficar)
    else:
        analisis_librosa(times, onset_env, onset_peaks, onset_detect, audio=audio, fs=fs, graficar=graficar)

    np.savetxt(
        salida,
        datos,
        header="tiempo_segundos,puntaje,clasificacion",
        delimiter=",",
        fmt=["%.4g", "%s", "%s"],
        comments="",
    )

    print(f"Ubicación de resultados: {salida}\n")

def contar_disparos_RF(audio, modelo="modelos/detector_disparos_RF.joblib", parametros_modelo="modelos/parametros_RF.json", salida="conteo_disparos.csv", umbral=0.5, prominencia=0.1, separacion=0.01, etiquetas_reales_csv=None):
    """
    Analiza un audio largo buscando picos de energía significativos mediante un umbral 
    de prominencia adaptativo, los clasifica usando el modelo entrenado y exporta 
    los resultados cronológicos a un archivo CSV.

    Args:
        audio (np.ndarray): Vector de muestras de audio largo que se desea auditar.
        salida (str): Ruta del archivo CSV donde se escribirá el reporte detallado.
        umbral (float): Probabilidad mínima (0.0 a 1.0) para validar un disparo positivo.
        prominencia (float): Porcentaje de la energía máxima requerido para detectar un pico de sonido.
        separacion (float): Tiempo de espera mínimo en segundos entre detecciones sucesivas.
        etiquetas_reales_csv (str): Ruta del archivo CSV con las etiquetas reales de disparos.
    
    Returns:
        None: No retorna ningún valor, pero escribe los resultados en el archivo CSV especificado.
    """
    modelo = joblib.load(modelo)
    parametros_RF = json.load(open(parametros_modelo))
    energia = energias(audio, parametros_RF["FRAME"])
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
    salida = Path(salida)
    with salida.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["tiempo_segundos", "puntaje", "clasificacion"])
        for indice in indices:
            muestra = min(int(indice) * parametros_RF["FRAME"] + parametros_RF["FRAME"] // 2, len(audio) - 1)
            x = caracteristicas(audio, muestra)
            puntaje = modelo.predict_proba([x])[0, 1] # Le solicita al Random Forest la probabilidad exacta (de 0.0 a 1.0) 
                                                      # de que este vector específico pertenezca a la clase positiva ("disparo" = 1)
            disparo = puntaje >= umbral
            total += int(disparo)
            # La siguiente línea inserta la primera fila de encabezados descriptivos 
            # en las columnas del reporte (tiempo_segundos, puntaje, clasificacion).
            escritor.writerow([f"{muestra / parametros_RF["FS"]:.3f}", f"{puntaje:.4f}",
                                "posible_disparo" if disparo else "otro"])
    
    print(f"--- Conteo de disparos con RandomForest ---")
    print(f"Candidatos a disparo: {len(indices)}")
    print(f"Cantidad de disparos detectados: {total}")
    if etiquetas_reales_csv is not None:
        etiquetas_reales = pd.read_csv(etiquetas_reales_csv)
        print(f"Cantidad de disparos efectuados: {len(etiquetas_reales)}")
    else:
        print("No se proporcionaron etiquetas reales.")
    print(f"Ubicación de resultados: {salida}\n")

def main():
    parser = argparse.ArgumentParser(description="Entrena un modelo de detección de disparos.")
    parser.add_argument("--audio", type=str, help="Ruta al archivo de audio")
    parser.add_argument("--salida", type=str, default="conteo_disparos.csv", help="Ruta para guardar el modelo entrenado")
    parser.add_argument("--etiquetas", type=str, default=None, help="Ruta a la carpeta de etiquetas reales")
    parser.add_argument("--metodo", type=str, default="librosa", help="Método de detección de disparos")
    parser.add_argument("--carpeta_modelo", type=str, default="modelos", help="Carpeta que contien al modelo RandomForest")
    parser.add_argument("--nombre_modelo", type=str, default="detector_disparos_RF.joblib", help="Nombre del modelo RandomForest")
    parser.add_argument("--umbral", type=float, default=0.5, help="Umbral de decisión para clasificar un disparo con RandomForest")
    parser.add_argument("--prominencia", type=float, default=0.1, help="Prominencia mínima para detectar un disparo con RandomForest")
    parser.add_argument("--separacion", type=float, default=0.01, help="Separación mínima entre disparos para RandomForest")
    parser.add_argument("--graficar", action="store_true", help="Indica si se deben graficar los resultados obtenidos con librosa")
    args = parser.parse_args()

    audio = args.audio
    salida_csv = args.salida
    etiquetas_reales_csv = args.etiquetas
    metodo = args.metodo
    if metodo == "RandomForest":
        modelo = Path(args.carpeta_modelo) / args.nombre_modelo
        parametros_RF = Path(args.carpeta_modelo) / "parametros_RF.json"
        contar_disparos_RF(audio, modelo=modelo, parametros_modelo=parametros_RF, salida=salida_csv, etiquetas_reales_csv=etiquetas_reales_csv,
                           umbral=args.umbral, prominencia=args.prominencia, separacion=args.separacion)
    elif metodo == "librosa":
        contar_disparos_librosa(audio, salida=salida_csv, etiquetas_reales_csv=etiquetas_reales_csv, graficar=args.graficar)
    else:
        print(f"Método de detección de disparos no reconocido: {metodo}")

if __name__ == "__main__":
    main()