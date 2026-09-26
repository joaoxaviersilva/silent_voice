from utils.llm_rerank import rerank


def main():
    context = (
        "A pessoa perguntou se você está com sede "
        "e ofereceu algo para beber."
    )

    candidates = [
        "AGUA",
        "COMER",
        "DORMIR",
    ]

    result = rerank(context, candidates)

    print("Contexto:", context)
    print("Candidatos:", candidates)
    print("Escolhida:", result)
    print("Resultado válido:", result in candidates)

    empty_result = rerank(context, [])

    print("Lista vazia:", empty_result)


if __name__ == "__main__":
    main()