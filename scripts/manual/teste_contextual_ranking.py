from fusion.contextual_ranking import rank_with_context


def print_ranking(title, ranking):
    print("=" * 70)
    print(title)
    print("=" * 70)

    for position, result in enumerate(
        ranking,
        start=1,
    ):
        print(
            f"{position}. {result['word']:<7} | "
            f"DTW: {result['original_distance']:.6f} | "
            f"Bônus: {result['context_bonus']:.6f} | "
            f"Ajustada: {result['adjusted_distance']:.6f}"
        )

    print()


def main():
    # Cenário 1:
    # QUEIJO venceu por uma diferença visual muito pequena.
    # Como PIZZA é mais relevante no contexto, ela pode assumir
    # a primeira posição.
    close_distances = {
        "QUEIJO": 0.0158,
        "PIZZA": 0.0160,
        "ARROZ": 0.0180,
    }

    food_context = [
        "PIZZA",
        "ARROZ",
        "QUEIJO",
        "HAMBÚRGUER",
    ]

    close_ranking = rank_with_context(
        close_distances,
        food_context,
    )

    print_ranking(
        "Cenário visualmente próximo",
        close_ranking,
    )

    # Cenário 2:
    # QUEIJO possui vantagem visual muito grande.
    # O contexto não pode fazer PIZZA vencer.
    clear_distances = {
        "QUEIJO": 0.0130,
        "PIZZA": 0.0190,
        "ARROZ": 0.0200,
    }

    clear_ranking = rank_with_context(
        clear_distances,
        food_context,
    )

    print_ranking(
        "Cenário visualmente claro",
        clear_ranking,
    )

    # Cenário 3:
    # Nenhuma palavra calibrada aparece no contexto.
    game_context = [
        "FUTEBOL",
        "XADREZ",
        "VIDEOGAME",
    ]

    no_match_ranking = rank_with_context(
        close_distances,
        game_context,
    )

    print_ranking(
        "Contexto sem palavras calibradas",
        no_match_ranking,
    )

    print(
        "Vencedora no cenário próximo:",
        close_ranking[0]["word"],
    )

    print(
        "Vencedora no cenário claro:",
        clear_ranking[0]["word"],
    )

    print(
        "Vencedora sem contexto compatível:",
        no_match_ranking[0]["word"],
    )


if __name__ == "__main__":
    main()