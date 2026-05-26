# teste_audio_salvar.py

import sounddevice as sd
from scipy.io.wavfile import write

fs = 48000

print("Fale algo por 5 segundos...")

audio = sd.rec(
    int(5 * fs),
    samplerate=fs,
    channels=1,
    dtype="float32",
    device=17,
)

sd.wait()

write("teste.wav", fs, audio)

print("Arquivo salvo.")
