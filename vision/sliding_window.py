import numpy as np


class SlidingWindow:
    """
    Divide uma sequência de features em janelas sobrepostas.
    """

    def __init__(self, window_size=12, step=4):
        self.window_size = window_size
        self.step = step

    def generate(self, sequence):
        """
        Parameters
        ----------
        sequence : np.ndarray
            Sequência de features (N x F)

        Returns
        -------
        list
            Lista de tuplas:
                (índice_inicial, janela)
        """

        sequence = np.asarray(sequence)

        if len(sequence) < self.window_size:
            return []

        windows = []

        for start in range(
            0,
            len(sequence) - self.window_size + 1,
            self.step,
        ):

            end = start + self.window_size

            window = sequence[start:end]

            windows.append((start, window))

        return windows

    def count_windows(self, sequence_length):
        """
        Retorna quantas janelas serão geradas.
        """

        if sequence_length < self.window_size:
            return 0

        return ((sequence_length - self.window_size) // self.step) + 1
