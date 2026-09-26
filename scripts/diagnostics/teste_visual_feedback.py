from utils.visual_feedback import success


def main():
    print("Exibindo feedback visual...")

    success(
        message="Frase reconhecida!",
        duration_ms=1500,
    )

    print("Feedback encerrado.")


if __name__ == "__main__":
    main()