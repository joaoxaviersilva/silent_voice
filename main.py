import cv2

from audio.stream_engine import StreamingAudio
from vision.detector import FaceMeshDetector
from vision.roi import crop_mouth
from vision.features import extract_sequence
from vision.sliding_window import SlidingWindow

from utils.multimodal_encoder import MultiModalEncoder
from vision.multimodal_matcher import MultiModalMatcher

from utils.llm_context import get_theme_and_vocab
from utils.llm_rerank import rerank

from utils.incremental_decoder import IncrementalDecoder
from utils.phrase_builder import build_phrase
from utils.phrase_memory import PhraseMemory
from utils.self_learning import SelfLearning

from fusion.rejection import reject
from utils.visual_feedback import success


def main():
    audio = StreamingAudio()
    detector = FaceMeshDetector()

    encoder = MultiModalEncoder()
    matcher = MultiModalMatcher(encoder)

    window = SlidingWindow()
    decoder = IncrementalDecoder()

    phrase_memory = PhraseMemory()
    word_memory = SelfLearning()

    cap = cv2.VideoCapture(0)

    while True:
        print("\n🎤 Fale uma frase...")
        text = audio.listen_once()
        print("Contexto:", text)

        theme, vocab = get_theme_and_vocab(text)
        if not vocab:
            continue

        vocab = word_memory.boost(vocab)

        all_landmarks = []
        frames = []

        print("👁️ Capturando...")

        for _ in range(80):
            ret, frame = cap.read()
            if not ret:
                continue

            pts = detector.extract(frame)
            if pts is None:
                continue

            all_landmarks.append(pts)
            frames.append(crop_mouth(frame, pts))

            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) == 27:
                break

        if len(all_landmarks) < 15:
            continue

        feats = extract_sequence(all_landmarks)
        windows = window.generate(feats)

        final_words = []
        partial_phrase = ""

        for start, w in windows:
            seg_frames = frames[start : start + len(w)]

            probs = matcher.match(seg_frames, vocab)

            if reject(probs):
                continue

            top = sorted(probs, key=probs.get, reverse=True)[:5]
            word = rerank(text, top)

            confirmed = decoder.update(word)

            if confirmed:
                final_words.append(confirmed)
                word_memory.add(confirmed)

                partial_phrase = " ".join(final_words)

                suggestion = phrase_memory.suggest(partial_phrase)
                if suggestion:
                    partial_phrase = suggestion

                print("🧠 Parcial:", partial_phrase)

        if not final_words:
            print("Nada detectado")
            continue

        phrase = build_phrase(text, final_words)

        print("✅ Frase final:", phrase)

        phrase_memory.add(phrase)
        success()


if __name__ == "__main__":
    main()
