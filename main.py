import cv2
import numpy as np

from audio.stream_engine import StreamingAudio
from fusion.contextual_ranking import rank_with_context
from utils.llm_context import get_theme_and_vocab
from utils.phrase_builder import build_phrase
from utils.phrase_memory import PhraseMemory
from utils.self_learning import SelfLearning
from utils.visual_feedback import success
from vision.detector import FaceMeshDetector
from vision.features import extract_sequence
from vision.temporal_matcher import TemporalLipMatcher


# ============================================================
# CONFIGURAÇÕES
# ============================================================

# Iriun Webcam
CAMERA_INDEX = 1

# Mesmo tamanho utilizado na coleta do dataset
CAPTURE_FRAMES = 40

# Quantidade mínima de frames válidos
MIN_VALID_FRAMES = 30

# Proteção contra loop infinito na captura
MAX_CAPTURE_ATTEMPTS = 100

# Maior distância DTW original aceita
MAX_DTW_DISTANCE = 0.020

# Rejeição de boca parada / resultado ambíguo
MIN_MOVEMENT_ENERGY = 0.0095
MIN_STATIC_MARGIN = 0.0005

# Quanto o contexto pode ajudar no máximo
MAX_CONTEXT_BONUS = 0.0006


# ============================================================
# INTERFACE VISUAL
# ============================================================


def draw_text(
    frame,
    text,
    vertical_position,
    font_scale=0.8,
    thickness=2,
):
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_size, _ = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    text_width = text_size[0]

    horizontal_position = (
        frame.shape[1] - text_width
    ) // 2

    cv2.putText(
        frame,
        text,
        (
            max(horizontal_position, 10),
            vertical_position,
        ),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_available_words(
    frame,
    available_words,
    start_y=95,
):
    words_per_line = 3

    for line_index in range(
        0,
        len(available_words),
        words_per_line,
    ):
        line_words = available_words[
            line_index : line_index + words_per_line
        ]

        line_text = " | ".join(
            line_words
        )

        vertical_position = (
            start_y
            + (line_index // words_per_line) * 35
        )

        draw_text(
            frame,
            line_text,
            vertical_position,
            font_scale=0.65,
        )


def wait_for_visual_start(
    cap,
    available_words,
):
    while True:
        ret, frame = cap.read()

        if not ret:
            print(
                "Aviso: não foi possível capturar "
                "a imagem da câmera."
            )
            continue

        draw_text(
            frame,
            "Prepare-se para a fala muda",
            50,
            font_scale=0.9,
        )

        draw_available_words(
            frame,
            available_words,
            start_y=95,
        )

        number_of_lines = max(
            1,
            (
                len(available_words) + 2
            )
            // 3,
        )

        instruction_y = (
            110
            + number_of_lines * 35
        )

        draw_text(
            frame,
            "ESPACO: iniciar | ESC: sair",
            instruction_y,
            font_scale=0.65,
        )

        cv2.imshow(
            "Silent Voice - Iriun Webcam",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            return False

        if key == 32:
            return True


def show_countdown(cap):
    for number in range(3, 0, -1):
        start_time = cv2.getTickCount()

        while True:
            ret, frame = cap.read()

            if not ret:
                return False

            draw_text(
                frame,
                "Prepare-se...",
                60,
                font_scale=1.0,
            )

            draw_text(
                frame,
                str(number),
                frame.shape[0] // 2,
                font_scale=4.0,
                thickness=6,
            )

            cv2.imshow(
                "Silent Voice - Iriun Webcam",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                return False

            elapsed_seconds = (
                cv2.getTickCount()
                - start_time
            ) / cv2.getTickFrequency()

            if elapsed_seconds >= 1:
                break

    return True


# ============================================================
# CAPTURA VISUAL
# ============================================================


def capture_visual_sequence(
    cap,
    detector,
):
    landmarks_sequence = []
    attempts = 0

    while (
        len(landmarks_sequence) < CAPTURE_FRAMES
        and attempts < MAX_CAPTURE_ATTEMPTS
    ):
        attempts += 1

        ret, frame = cap.read()

        if not ret:
            continue

        landmarks = detector.extract(
            frame
        )

        if landmarks is not None:
            landmarks_sequence.append(
                landmarks
            )

        valid_frames = len(
            landmarks_sequence
        )

        progress = (
            valid_frames
            / CAPTURE_FRAMES
        )

        progress_width = int(
            frame.shape[1]
            * progress
        )

        cv2.rectangle(
            frame,
            (
                0,
                frame.shape[0] - 20,
            ),
            (
                progress_width,
                frame.shape[0],
            ),
            (0, 255, 0),
            -1,
        )

        draw_text(
            frame,
            "ARTICULE UMA PALAVRA UMA VEZ",
            50,
            font_scale=0.85,
        )

        draw_text(
            frame,
            (
                f"Frames: "
                f"{valid_frames}/"
                f"{CAPTURE_FRAMES}"
            ),
            95,
            font_scale=0.7,
        )

        cv2.imshow(
            "Silent Voice - Iriun Webcam",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            return None, True

    if (
        len(landmarks_sequence)
        < MIN_VALID_FRAMES
    ):
        return None, False

    return landmarks_sequence, False


# ============================================================
# ANÁLISE
# ============================================================


def calculate_movement_energy(
    features,
):
    if (
        features.ndim != 2
        or features.shape[1] < 2
    ):
        return 0.0

    half = (
        features.shape[1] // 2
    )

    movement_features = features[
        :,
        half:,
    ]

    return float(
        np.mean(
            np.abs(
                movement_features
            )
        )
    )


def print_word_list(
    title,
    words,
):
    print(f"\n{title}")

    if not words:
        print("- Nenhuma")
        return

    for index, word in enumerate(
        words,
        start=1,
    ):
        print(
            f"{index:02d}. {word}"
        )


def print_visual_ranking(
    ranking,
):
    print(
        "\nRanking visual original (DTW):"
    )

    for position, (
        word,
        distance,
    ) in enumerate(
        ranking,
        start=1,
    ):
        print(
            f"{position}. "
            f"{word}: "
            f"{distance:.6f}"
        )


def print_contextual_ranking(
    ranking,
):
    print(
        "\nRanking após preferência contextual:"
    )

    for position, result in enumerate(
        ranking,
        start=1,
    ):
        word = result["word"]

        original_distance = result[
            "original_distance"
        ]

        context_bonus = result[
            "context_bonus"
        ]

        adjusted_distance = result[
            "adjusted_distance"
        ]

        print(
            f"{position}. "
            f"{word:<10} | "
            f"DTW: "
            f"{original_distance:.6f} | "
            f"Bônus: "
            f"{context_bonus:.6f} | "
            f"Ajustada: "
            f"{adjusted_distance:.6f}"
        )


# ============================================================
# MAIN
# ============================================================


def main():
    print("=" * 60)
    print(
        "Inicializando o Silent Voice..."
    )
    print("=" * 60)

    # --------------------------------------------------------
    # ÁUDIO
    # --------------------------------------------------------

    print(
        "\nInicializando o áudio..."
    )

    audio = StreamingAudio()

    # --------------------------------------------------------
    # VISÃO
    # --------------------------------------------------------

    print(
        "\nInicializando o detector facial..."
    )

    detector = FaceMeshDetector()

    # --------------------------------------------------------
    # MATCHER TEMPORAL
    # --------------------------------------------------------

    print(
        "\nCarregando as amostras temporais..."
    )

    temporal_matcher = (
        TemporalLipMatcher(
            dataset_path=(
                "data/lip_samples"
            ),
        )
    )

    dataset_summary = (
        temporal_matcher.dataset_summary()
    )

    calibrated_words = sorted(
        dataset_summary.keys()
    )

    if not calibrated_words:
        print(
            "Erro: nenhuma palavra "
            "calibrada foi encontrada."
        )
        return

    print(
        "\nPalavras calibradas:"
    )

    for word in calibrated_words:
        quantity = (
            dataset_summary[word]
        )

        print(
            f"- {word}: "
            f"{quantity} amostras"
        )

    # --------------------------------------------------------
    # MEMÓRIAS
    # --------------------------------------------------------

    phrase_memory = PhraseMemory()
    word_memory = SelfLearning()

    # --------------------------------------------------------
    # CÂMERA
    # --------------------------------------------------------

    print(
        "\nAbrindo a Iriun Webcam..."
    )

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW,
    )

    if not cap.isOpened():
        print(
            "Erro: não foi possível "
            "abrir a Iriun Webcam."
        )

        print(
            "Confirme se o Iriun está "
            "aberto no celular e no PC."
        )

        return

    print(
        "Iriun Webcam aberta "
        "com sucesso."
    )

    # ========================================================
    # LOOP PRINCIPAL
    # ========================================================

    try:
        while True:
            print(
                "\n"
                + "=" * 60
            )

            print(
                "🎤 Fale uma frase para "
                "fornecer o contexto..."
            )

            print(
                "=" * 60
            )

            # ------------------------------------------------
            # CONTEXTO DE ÁUDIO
            # ------------------------------------------------

            text = audio.listen_once()

            text = str(
                text or ""
            ).strip()

            if not text:
                print(
                    "Nenhum contexto de áudio "
                    "foi identificado."
                )
                continue

            print(
                "\nContexto reconhecido:"
            )

            print(text)

            # ------------------------------------------------
            # CONTEXTO SEMÂNTICO
            # ------------------------------------------------

            print(
                "\nGerando tema e "
                "vocabulário contextual..."
            )

            (
                theme,
                contextual_vocab,
            ) = get_theme_and_vocab(
                text
            )

            contextual_vocab = (
                contextual_vocab or []
            )

            print(
                "\nTema:"
            )

            print(
                theme
                or "Não identificado"
            )

            print_word_list(
                (
                    "Vocabulário sugerido "
                    "pelo contexto:"
                ),
                contextual_vocab,
            )

            # ------------------------------------------------
            # PALAVRAS CALIBRADAS
            # ------------------------------------------------

            comparison_words = list(
                calibrated_words
            )

            print_word_list(
                (
                    "Palavras disponíveis "
                    "para reconhecimento:"
                ),
                comparison_words,
            )

            print(
                "\nTodas as palavras calibradas "
                "continuam disponíveis."
            )

            print(
                "O contexto apenas fornece "
                "uma pequena preferência."
            )

            # ------------------------------------------------
            # PREPARAÇÃO DA FALA MUDA
            # ------------------------------------------------

            print(
                "\nPosicione a câmera "
                "próxima da boca."
            )

            print(
                "Pressione ESPAÇO quando "
                "estiver pronto."
            )

            start_requested = (
                wait_for_visual_start(
                    cap,
                    comparison_words,
                )
            )

            if not start_requested:
                print(
                    "\nEncerrando o "
                    "Silent Voice..."
                )
                break

            # ------------------------------------------------
            # CONTAGEM REGRESSIVA
            # ------------------------------------------------

            countdown_completed = (
                show_countdown(
                    cap
                )
            )

            if not countdown_completed:
                print(
                    "\nEncerrando o "
                    "Silent Voice..."
                )
                break

            # ------------------------------------------------
            # CAPTURA
            # ------------------------------------------------

            (
                landmarks_sequence,
                exit_requested,
            ) = capture_visual_sequence(
                cap,
                detector,
            )

            if exit_requested:
                print(
                    "\nEncerrando o "
                    "Silent Voice..."
                )
                break

            if landmarks_sequence is None:
                print(
                    "Não foram capturados "
                    "frames válidos "
                    "suficientes."
                )
                continue

            # ------------------------------------------------
            # FEATURES
            # ------------------------------------------------

            print(
                "\nProcessando o movimento "
                "dos lábios..."
            )

            features = extract_sequence(
                landmarks_sequence
            )

            print(
                f"Shape capturado: "
                f"{features.shape}"
            )

            movement_energy = (
                calculate_movement_energy(
                    features
                )
            )

            # ------------------------------------------------
            # DTW
            # ------------------------------------------------

            (
                _,
                distances,
            ) = temporal_matcher.predict(
                features,
                allowed_words=(
                    comparison_words
                ),
            )

            if not distances:
                print(
                    "Nenhuma comparação "
                    "temporal foi gerada."
                )
                continue

            # ------------------------------------------------
            # RANKING VISUAL ORIGINAL
            # ------------------------------------------------

            visual_ranking = sorted(
                distances.items(),
                key=lambda item: item[1],
            )

            print_visual_ranking(
                visual_ranking
            )

            visual_best_word = (
                visual_ranking[0][0]
            )

            visual_best_distance = (
                visual_ranking[0][1]
            )

            if len(visual_ranking) >= 2:
                visual_second_distance = (
                    visual_ranking[1][1]
                )

                visual_margin = (
                    visual_second_distance
                    - visual_best_distance
                )
            else:
                visual_margin = 0.0

            # ------------------------------------------------
            # PREFERÊNCIA CONTEXTUAL
            # ------------------------------------------------

            contextual_ranking = (
                rank_with_context(
                    distances=distances,
                    contextual_vocab=(
                        contextual_vocab
                    ),
                    maximum_context_bonus=(
                        MAX_CONTEXT_BONUS
                    ),
                )
            )

            if not contextual_ranking:
                print(
                    "Não foi possível gerar "
                    "o ranking contextual."
                )
                continue

            print_contextual_ranking(
                contextual_ranking
            )

            selected_result = (
                contextual_ranking[0]
            )

            selected_word = (
                selected_result["word"]
            )

            selected_original_distance = (
                selected_result[
                    "original_distance"
                ]
            )

            selected_adjusted_distance = (
                selected_result[
                    "adjusted_distance"
                ]
            )

            selected_context_bonus = (
                selected_result[
                    "context_bonus"
                ]
            )

            # ------------------------------------------------
            # DIAGNÓSTICO
            # ------------------------------------------------

            print(
                f"\nEnergia de movimento: "
                f"{movement_energy:.6f}"
            )

            print(
                f"Melhor palavra visual: "
                f"{visual_best_word}"
            )

            print(
                f"Margem visual: "
                f"{visual_margin:.6f}"
            )

            print(
                f"Palavra após contexto: "
                f"{selected_word}"
            )

            print(
                f"Distância DTW original "
                f"da escolhida: "
                f"{selected_original_distance:.6f}"
            )

            print(
                f"Bônus contextual: "
                f"{selected_context_bonus:.6f}"
            )

            print(
                f"Distância ajustada: "
                f"{selected_adjusted_distance:.6f}"
            )

            # ------------------------------------------------
            # REJEIÇÃO
            # ------------------------------------------------

            rejection_reasons = []

            # Detecta baixa movimentação acompanhada
            # de forte ambiguidade visual.
            #
            # IMPORTANTE:
            # usamos a margem VISUAL ORIGINAL,
            # e não a margem alterada pelo contexto.
            if (
                movement_energy
                < MIN_MOVEMENT_ENERGY
                and visual_margin
                < MIN_STATIC_MARGIN
            ):
                rejection_reasons.append(
                    (
                        "movimento labial "
                        "insuficiente e "
                        "resultado visual ambíguo"
                    )
                )

            # O contexto nunca pode salvar uma
            # palavra visualmente distante demais.
            if (
                selected_original_distance
                > MAX_DTW_DISTANCE
            ):
                rejection_reasons.append(
                    (
                        "distância temporal "
                        "original acima do limite"
                    )
                )

            if rejection_reasons:
                print(
                    "\n❌ Movimento rejeitado."
                )

                for reason in (
                    rejection_reasons
                ):
                    print(
                        f"- {reason}"
                    )

                print(
                    "Articule uma das palavras "
                    "cadastradas e tente novamente."
                )

                continue

            # ------------------------------------------------
            # RESULTADO
            # ------------------------------------------------

            if (
                selected_word
                != visual_best_word
            ):
                print(
                    "\n🧠 O contexto alterou "
                    "a preferência do ranking."
                )

                print(
                    f"Visual: "
                    f"{visual_best_word}"
                )

                print(
                    f"Com contexto: "
                    f"{selected_word}"
                )

            print(
                "\n✅ Palavra reconhecida:",
                selected_word,
            )

            # ------------------------------------------------
            # MEMÓRIA
            # ------------------------------------------------

            word_memory.add(
                selected_word
            )

            # ------------------------------------------------
            # FRASE FINAL
            # ------------------------------------------------

            phrase = build_phrase(
                text,
                [selected_word],
            )

            if not phrase:
                phrase = selected_word

            print(
                "✅ Frase final:",
                phrase,
            )

            phrase_memory.add(
                phrase
            )

            success(
                message=(
                    "Palavra reconhecida!"
                ),
                duration_ms=1500,
            )

    except KeyboardInterrupt:
        print(
            "\nExecução interrompida "
            "pelo usuário."
        )

    except Exception as error:
        print(
            f"\nErro durante a execução: "
            f"{error}"
        )

    finally:
        cap.release()
        cv2.destroyAllWindows()

        print(
            "Câmera e janelas encerradas."
        )


if __name__ == "__main__":
    main()