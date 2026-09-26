class IncrementalDecoder:
    def __init__(self, stability_threshold=3):
        if stability_threshold < 1:
            raise ValueError(
                "O limite de estabilidade deve ser maior que zero."
            )

        self.threshold = stability_threshold
        self.current_word = None
        self.count = 0
        self.locked_word = None

    def update(self, word):
        # Uma previsão rejeitada interrompe a sequência atual.
        if not word:
            self.reset()
            return None

        # Evita confirmar várias vezes a mesma palavra
        # enquanto ela continuar sendo detectada.
        if word == self.locked_word:
            return None

        # Ao detectar outra palavra, libera a anterior.
        self.locked_word = None

        if word == self.current_word:
            self.count += 1
        else:
            self.current_word = word
            self.count = 1

        if self.count < self.threshold:
            return None

        confirmed_word = self.current_word
        self.locked_word = confirmed_word

        self.current_word = None
        self.count = 0

        return confirmed_word

    def reset(self):
        self.current_word = None
        self.count = 0
        self.locked_word = None