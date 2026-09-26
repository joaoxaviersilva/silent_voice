from config import CONF_THRESHOLD


def decide(probabilities, threshold=CONF_THRESHOLD):
    if not probabilities:
        return None

    best_word, best_probability = max(
        probabilities.items(),
        key=lambda item: item[1],
    )

    if best_probability >= threshold:
        return best_word

    return None