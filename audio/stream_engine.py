import sounddevice as sd
import whisper

from scipy.signal import resample


class StreamingAudio:
    def __init__(self):
        print("Carregando modelo Whisper...")
        self.model = whisper.load_model("small")

    def listen_once(self, duration=5):
        samplerate = 48000

        print("🎤 Gravando...")

        audio = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            device=17,
        )

        sd.wait()

        audio = audio.flatten()

        print("Shape:", audio.shape)
        print("Tipo:", audio.dtype)
        print("Min:", float(audio.min()))
        print("Max:", float(audio.max()))

        # converte 48k -> 16k
        target_rate = 16000

        audio = resample(audio, int(len(audio) * target_rate / samplerate))

        print("🧠 Transcrevendo...")

        result = self.model.transcribe(
            audio,
            language="pt",
            fp16=False,
        )

        return result["text"].lower()
