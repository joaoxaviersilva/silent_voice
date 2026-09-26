from pygrabber.dshow_graph import FilterGraph


def main():
    devices = FilterGraph().get_input_devices()

    if not devices:
        print("Nenhuma câmera encontrada.")
        return

    print("Câmeras encontradas:")

    for index, name in enumerate(devices):
        print(f"{index}: {name}")


if __name__ == "__main__":
    main()