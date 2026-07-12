import numpy as np

from vision.sliding_window import SlidingWindow


def main():

    # Simula 80 frames
    features = np.random.rand(80, 6)

    window = SlidingWindow(
        window_size=12,
        step=4,
    )

    windows = window.generate(features)

    print("=" * 50)
    print("Frames:", len(features))
    print("Janelas:", len(windows))
    print("=" * 50)

    for i, (start, data) in enumerate(windows[:5]):

        print(f"Janela {i+1}")
        print("Início:", start)
        print("Shape :", data.shape)
        print()


if __name__ == "__main__":
    main()