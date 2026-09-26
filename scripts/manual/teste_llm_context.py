from utils.llm_context import get_theme_and_vocab


def main():
    text = "Você está com sede? Quer beber alguma coisa?"

    theme, vocab = get_theme_and_vocab(text)

    print("Texto:", text)
    print("Tema:", theme)
    print("Vocabulário:", vocab)
    print("Quantidade:", len(vocab))
    print("Máximo de 15:", len(vocab) <= 15)
    print("Sem duplicados:", len(vocab) == len(set(vocab)))
    print(
        "Todas em maiúsculas:",
        all(word == word.upper() for word in vocab),
    )

    empty_theme, empty_vocab = get_theme_and_vocab("")

    print("\nTexto vazio:")
    print("Tema:", repr(empty_theme))
    print("Vocabulário:", empty_vocab)


if __name__ == "__main__":
    main()