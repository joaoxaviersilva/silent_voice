import whisper

model = whisper.load_model("small")

result = model.transcribe(
    "teste.wav",
    language="pt",
    fp16=False,
)

print(result["text"])
