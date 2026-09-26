from fusion.rejection import normalized_entropy, reject


def show_result(name, probabilities):
    uncertainty = normalized_entropy(probabilities)

    print(name)
    print(f"Entropia normalizada: {uncertainty:.4f}")
    print(f"Rejeitado: {reject(probabilities)}")
    print()


def main():
    uncertain_result = {
        "agua": 0.3547,
        "comer": 0.3284,
        "bom dia": 0.3169,
    }

    confident_result = {
        "agua": 0.90,
        "comer": 0.06,
        "bom dia": 0.04,
    }

    show_result("Resultado incerto", uncertain_result)
    show_result("Resultado confiante", confident_result)


if __name__ == "__main__":
    main()