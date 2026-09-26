import numpy as np


EPSILON = 1e-12


def entropy(probabilities):
    if not probabilities:
        return 0.0

    values = np.asarray(
        list(probabilities.values()),
        dtype=np.float64,
    )

    values = np.clip(values, 0.0, None)

    total = np.sum(values)

    if total <= 0:
        return 0.0

    values = values / total

    return float(
        -np.sum(values * np.log(values + EPSILON))
    )


def normalized_entropy(probabilities):
    number_of_classes = len(probabilities)

    if number_of_classes <= 1:
        return 0.0

    maximum_entropy = np.log(number_of_classes)

    return entropy(probabilities) / maximum_entropy


def reject(probabilities, threshold=0.90):
    if not probabilities:
        return True

    uncertainty = normalized_entropy(probabilities)

    return uncertainty >= threshold