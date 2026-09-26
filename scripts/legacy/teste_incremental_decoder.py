from utils.incremental_decoder import IncrementalDecoder


def test_sequence(name, sequence):
    decoder = IncrementalDecoder(stability_threshold=3)

    print(name)

    for word in sequence:
        confirmed = decoder.update(word)
        print(f"Entrada: {word!r} | Confirmada: {confirmed!r}")

    print()


def main():
    test_sequence(
        "Sequência estável",
        ["agua", "agua", "agua"],
    )

    test_sequence(
        "Sequência instável",
        ["agua", "comer", "agua"],
    )

    test_sequence(
        "Sequência interrompida",
        ["agua", "agua", None, "agua"],
    )

    test_sequence(
        "Palavra repetida em seis janelas",
        ["agua", "agua", "agua", "agua", "agua", "agua"],
    )


if __name__ == "__main__":
    main()