import matplotlib.pyplot as plt
import numpy as np

import librosa

# Carga de archivo
archivo = "audios/Audio 1 - Recorte Damian Kuc - Recortado.mp3"
y, sr = librosa.load(archivo, sr=None)

# Short-Time Fourier Transform (STFT)
stft = librosa.stft(y)

# Cambio de unidades a dB y calculo de curva de novedad
logS = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
onset_env = librosa.onset.onset_strength(S=logS)

# Filtrado de maximos de la curva de novedad para obtener disparos
times = librosa.times_like(onset_env, sr=sr)
onset_peaks = librosa.util.localmax(onset_env)
onset_detect = librosa.onset.onset_detect(onset_envelope=onset_env)

print(f"Cantidad de disparos detectados: {len(onset_detect)}")
datos = np.empty((len(onset_detect), 2), dtype=object)
datos[:, 0] = times[onset_detect]
datos[:, 1] = 'posible_disparo'

np.savetxt(
    "extraccion_disparos.csv",
    datos,
    header="tiempo_segundos,clasificacion",
    delimiter=",",
    fmt=["%.4g", "%s"],
)