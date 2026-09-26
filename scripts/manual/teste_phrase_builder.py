from utils.phrase_builder import build_phrase


def main():
    context = "A pessoa perguntou se você deseja beber alguma coisa."
    words = ["EU", "QUERO", "ÁGUA"]

    phrase = build_phrase(context, words)

    print("Contexto:", context)
    print("Palavras:", words)
    print("Frase construída:", phrase)
    print("Frase válida:", isinstance(phrase, str) and bool(phrase.strip()))

    empty_result = build_phrase(context, [])

    print("\nLista vazia:")
    print("Resultado:", repr(empty_result))


if __name__ == "__main__":
    main()