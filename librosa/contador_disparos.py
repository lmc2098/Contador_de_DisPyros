import numpy as np
import matplotlib.pyplot as plt

import librosa

archivo = 'audios/Audio 1 - Recorte Damian Kuc.mp3'
y, sr = librosa.load(archivo,sr=None)

stft = librosa.stft(y)

logS = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
onset_env = librosa.onset.onset_strength(S=logS)

times = librosa.times_like(onset_env, sr=sr)
onset_peaks = librosa.util.localmax(onset_env)
onset_detect = librosa.onset.onset_detect(onset_envelope=onset_env)

fig, ax = plt.subplots(nrows=2, sharex=True, height_ratios=(3, 1))

librosa.display.waveshow(y=y, sr=sr, ax=ax[1], label="Waveform")
ax[1].legend()
ax[0].plot(times, onset_env, label="Onset envelope", color="C1")
ax[0].scatter(times[onset_peaks], onset_env[onset_peaks], marker="^", color="k", label="Localmax Peaks")
ax[0].scatter(times[onset_detect], onset_env[onset_detect], marker="o",
              edgecolor="C2", facecolor="none", label="onset_detect")
ax[0].legend()
ax[0].label_outer()
plt.show()