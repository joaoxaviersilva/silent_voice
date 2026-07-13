from audio.stream_engine import StreamingAudio

audio = StreamingAudio()

print("Fale algo...")
texto = audio.listen_once(duration=5)

print("Resultado:")
print(texto)
