
import pickle


def read_data(target_data):
    try:
        with open(target_data, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        raise ValueError("Error: picle file not found.")

    print("Data shpe:", data.shape)

    return data
