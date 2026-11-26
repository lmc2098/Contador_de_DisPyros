import os
import glob
import numpy as np
import math
import librosa
from scipy.io import wavfile
from scipy.signal import resample_poly

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

def duracion_disparo_promedio(carpeta_path, threshold_db=-30):
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

def leer_wav(ruta, fs_objetivo=16000):
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
    fs_original, audio = wavfile.read(ruta)
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
    if fs_original != fs_objetivo:
        audio = resample_poly(audio, fs_objetivo, fs_original)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError(f"Audio inválido: {ruta}")
    if fs_original != fs_objetivo:
        # Si la frecuencia de muestreo original no es igual a la frecuencia objetivo (FS), 
        # calcula el Máximo Común Divisor y aplica un filtrado y rediseño polifásico (resample_poly) 
        # para cambiar el muestreo sin deformar el tono del audio.
        divisor = math.gcd(fs_original, fs_objetivo)
        audio = resample_poly(audio, fs_objetivo // divisor, fs_original // divisor)
    return audio


def energias(audio, frame):
    """
    Divide la señal de audio en ventanas de tamaño FRAME y calcula
    la energía RMS (Root Mean Square) para cada una de ellas.

    Args:
        audio (np.ndarray): Vector de muestras de audio normalizado.

    Returns:
        np.ndarray: Vector con los niveles de energía calculados por cuadro.
    """
    relleno = (-len(audio)) % frame
    cuadros = np.pad(audio, (0, relleno)).reshape(-1, frame) # Rellena con ceros el final del audio y lo reestructura en una matriz cuyas 
                                                             # filas son ventanas de FRAME muestras.
    return np.sqrt(np.mean(cuadros ** 2, axis=1))


def recorte(audio, muestra, t_antes, t_despues):
    """
    Extrae un segmento de audio centrado en una muestra específica, aplicando 
    márgenes de tiempo especificados (t_antes y t_despues) y rellenando con ceros si el 
    recorte excede los límites físicos del archivo.

    Args:
        audio (np.ndarray): Vector completo de muestras de audio.
        muestra (int): Índice de la muestra central donde ocurre el evento.

    Returns:
        np.ndarray: Segmento de audio aislado con una longitud fija de muestras.
    """
    inicio = muestra - t_antes
    fin = muestra + t_despues
    izquierda = max(0, -inicio)
    derecha = max(0, fin - len(audio))
    return np.pad(audio[max(0, inicio):min(len(audio), fin)],
                  (izquierda, derecha))


