import numpy as np
import pathlib


def read_flo(path: str | bytes | pathlib.Path) -> np.ndarray:
    with open(path, 'rb') as file:
        assert np.fromfile(file, np.float32, count=1).item() == 202021.25, "Incorrect magic number in flo file"
        width, height = np.fromfile(file, np.int32, count=2)
        return np.fromfile(file, np.float32, count=2 * width * height).reshape(height, width, 2)


def save_flo(data: np.ndarray, path: str | bytes | pathlib.Path) -> None:
    with open(path, 'wb') as file:
        file.write("PIEH".encode("ascii"))
        _, components, height, width = data.shape
        assert components == 2
        data.reshape((height, width, components))
        file.write(width.to_bytes(4) + height.to_bytes(4))
        file.write(data.tobytes())
