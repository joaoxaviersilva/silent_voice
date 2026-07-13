import sounddevice as sd
import torch
import whisper
from scipy.signal import resample


class StreamingAudio:
    def __init__(self):
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Carregando modelo Whisper no dispositivo: {self.device_type}...")

        self.model = whisper.load_model("small", device=self.device_type)

    def listen_once(self, duration=5, device=None, samplerate=16000):
        """Grava e transcreve um áudio de forma dinâmica."""
        print("🎤 Gravando...")

        try:

            audio = sd.rec(
                int(duration * samplerate),
                samplerate=samplerate,
                channels=1,
                dtype="float32",
                device=device,
            )
            sd.wait()
        except Exception as e:
            print(f"⚠️ Erro ao gravar com o device selecionado ({device}): {e}")
            print("Tentando usar o dispositivo padrão de captura...")

            audio = sd.rec(
                int(duration * samplerate),
                samplerate=samplerate,
                channels=1,
                dtype="float32",
                device=None,
            )
            sd.wait()

        audio = audio.flatten()

        print("Shape:", audio.shape)
        print("Tipo:", audio.dtype)
        print("Min:", float(audio.min()))
        print("Max:", float(audio.max()))

        if samplerate != 16000:
            print(f"🔄 Convertendo {samplerate}Hz -> 16000Hz...")
            target_rate = 16000
            audio = resample(audio, int(len(audio) * target_rate / samplerate))

        print("🧠 Transcrevendo...")
        result = self.model.transcribe(
            audio,
            language="pt",
            fp16=False,
        )

        return result["text"].lower()
