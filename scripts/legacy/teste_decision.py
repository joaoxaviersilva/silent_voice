from fusion.decision import decide


def main():
    empty_result = {}

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

    exact_threshold_result = {
        "agua": 0.85,
        "comer": 0.10,
        "bom dia": 0.05,
    }

    print("Vazio:", decide(empty_result))
    print("Incerto:", decide(uncertain_result))
    print("Confiante:", decide(confident_result))
    print("No limiar:", decide(exact_threshold_result))


if __name__ == "__main__":
    main()