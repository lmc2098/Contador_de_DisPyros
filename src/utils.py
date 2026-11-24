import os
import glob
import numpy as np
import librosa

def calcular_duracion_disparo(file_path, threshold_db=-30):
    """
    Calcula la duración exacta de un disparo eliminando el pre y post-audio
    basándose en un umbral de energía RMS en decibelios.

    Utilizado para el cálculo de la duración promedio de un disparo para el 
    análisis posterior de un audio con 'librosa' para detección de disparos.

    Args:
        file_path (str): Ruta del archivo de audio WAV a analizar.
        threshold_db (float): Umbral de energía RMS en decibelios para considerar un disparo.

    Returns:
        duracion_real (float): Duración real del disparo en segundos. 
                               Retorna 0 si no se detecta ningún disparo.
    """
    # Carga el audio (en formato 'mono' para simplificar el análisis de amplitud)
    y, sr = librosa.load(file_path, sr=None, mono=True)
    
    # Calcular la energía RMS por cuadros (frames)
    hop_length = 128  # Ventana pequeña (en cantidad de muestras) para alta precisión temporal
    rms = librosa.feature.rms(y=y, frame_length=256, hop_length=hop_length)[0]
    
    # Conversión de la amplitud RMS a Decibelios (dB) relativos al pico máximo
    # El pico máximo del disparo será 0 dB.
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # Búsqueda de los índices que superan el umbral (donde realmente suena el disparo)
    indices_disparo = np.where(rms_db > threshold_db)[0]
    
    if len(indices_disparo) == 0:
        # Si nada supera el umbral, retorna 0
        return 0.0
    
    # Obtener el primer y último cuadro del disparo (Onset y Offset)
    frame_inicio = indices_disparo[0]
    frame_fin = indices_disparo[-1]
    
    # Convertir los cuadros a tiempo real (segundos)
    tiempo_inicio = librosa.frames_to_time(frame_inicio, sr=sr, hop_length=hop_length)
    tiempo_fin = librosa.frames_to_time(frame_fin, sr=sr, hop_length=hop_length)
    
    duracion_real = tiempo_fin - tiempo_inicio
    return duracion_real

def procesar_carpeta_disparos(carpeta_path, threshold_db=-30):
    """ 
    Buscar todos los archivos .wav en la ruta especificada y calcular la duración
    de los disparos utilizando un umbral de energía RMS en decibelios.

    Imprime los resultados del análisis de los disparos encontrados en la carpeta.

    Args:
        carpeta_path (str): Ruta de la carpeta que contiene los archivos .wav.
        threshold_db (float): Umbral de energía RMS en decibelios para considerar un disparo.

    Returns:
        None
    """
    archivos = glob.glob(os.path.join(carpeta_path, "*.wav"))
    
    if not archivos:
        print(f"No se encontraron archivos .wav en la ruta: {carpeta_path}")
        return
    
    duraciones = []
    
    for f in archivos:
        duracion = calcular_duracion_disparo(f, threshold_db)
        if duracion > 0:
            duraciones.append(duracion)
            
    if duraciones:
        promedio = np.mean(duraciones)
        desviacion = np.std(duraciones)
        print("=" * 45)
        print(f" RESULTADOS DEL ANÁLISIS DE DISPAROS")
        print("=" * 45)
        print(f"Total de archivos procesados: {len(duraciones)}")
        print(f"Duración promedio real:       {promedio:.4f} segundos")
        print(f"Desviación estándar:          {desviacion:.4f} segundos")
        print(f"Disparo más corto detectado:  {np.min(duraciones):.4f} s")
        print(f"Disparo más largo detectado:  {np.max(duraciones):.4f} s")
        print("=" * 45)
    else:
        print("No se pudo extraer la duración de ningún disparo con el umbral actual.")


