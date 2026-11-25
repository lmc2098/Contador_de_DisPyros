import matplotlib.pyplot as plt
import numpy as np

import librosa

# Carga de archivo de audio
archivo = "audios/Audio 1 - Recorte Damian Kuc - Recortado.mp3"
y, sr = librosa.load(archivo, sr=None)

# Carga de archivo con etiquetas de disparos detectados manualmente
etiquetas = np.loadtxt("audios/Masacre de Ramallo - Test/Audio 1 - Etiquetas.txt", dtype=str, delimiter="\t")
etiqueta = np.array([float(x) for x in etiquetas[:,0]])

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
print(f"Cantidad de disparos efectuados: {len(etiqueta)}")

# Visualizacion de resultados
fig, ax = plt.subplots(nrows=2, sharex=True, height_ratios=(3, 1))

librosa.display.waveshow(y=y, sr=sr, ax=ax[1], label="Waveform")
ax[1].legend()
ax[0].plot(times, onset_env, label="Onset envelope", color="C1")
ax[0].scatter(
    times[onset_peaks],
    onset_env[onset_peaks],
    marker="^",
    color="k",
    label="Localmax Peaks",
)
ax[0].scatter(
    times[onset_detect],
    onset_env[onset_detect],
    marker="o",
    edgecolor="C2",
    facecolor="none",
    label="Disparos detectados",
)
ax[0].vlines(
    etiqueta,
    ymin=0,
    ymax=max(onset_env[onset_peaks]),
    color='red',
    linestyle='dashed',
    linewidth=1,
    label='Disparos Efectuados'
)
ax[0].legend()
ax[0].label_outer()
plt.show()
