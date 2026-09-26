def normalize_word(word):
    """
    Padroniza uma palavra para comparação.
    """

    return str(word).strip().upper()


def prepare_contextual_vocab(contextual_vocab):
    """
    Normaliza o vocabulário contextual e remove duplicados,
    preservando a ordem fornecida pelo LLM.
    """

    normalized_vocab = []

    for word in contextual_vocab or []:
        normalized_word = normalize_word(word)

        if not normalized_word:
            continue

        if normalized_word in normalized_vocab:
            continue

        normalized_vocab.append(normalized_word)

    return normalized_vocab


def rank_with_context(
    distances,
    contextual_vocab,
    maximum_context_bonus=0.0006,
):
    """
    Ajusta as distâncias do DTW utilizando o contexto.

    Quanto menor a distância, melhor.

    Uma palavra presente no contexto recebe um pequeno bônus,
    reduzindo sua distância ajustada. O bônus é maior para as
    primeiras palavras sugeridas pelo LLM.

    O contexto apenas ajuda em resultados próximos. Uma diferença
    visual grande continua prevalecendo.

    Args:
        distances:
            Dicionário no formato:
            {"PIZZA": 0.016, "ARROZ": 0.018}

        contextual_vocab:
            Palavras sugeridas pelo LLM, em ordem de relevância.

        maximum_context_bonus:
            Maior redução permitida na distância.

    Returns:
        Lista ordenada de dicionários contendo os detalhes
        do ranking final.
    """

    if not distances:
        return []

    if maximum_context_bonus < 0:
        raise ValueError(
            "O bônus contextual não pode ser negativo."
        )

    normalized_context = prepare_contextual_vocab(
        contextual_vocab
    )

    contextual_positions = {
        word: position
        for position, word in enumerate(
            normalized_context
        )
    }

    ranking = []

    for word, distance in distances.items():
        normalized_word = normalize_word(word)
        original_distance = float(distance)

        context_bonus = 0.0
        context_position = None

        if normalized_word in contextual_positions:
            context_position = contextual_positions[
                normalized_word
            ]

            # Primeira palavra recebe o bônus completo.
            # Segunda recebe metade, terceira recebe 1/3 etc.
            context_bonus = (
                maximum_context_bonus
                / (context_position + 1)
            )

        adjusted_distance = max(
            0.0,
            original_distance - context_bonus,
        )

        ranking.append(
            {
                "word": normalized_word,
                "original_distance": original_distance,
                "context_bonus": context_bonus,
                "adjusted_distance": adjusted_distance,
                "context_position": context_position,
            }
        )

    ranking.sort(
        key=lambda item: item["adjusted_distance"]
    )

    return ranking