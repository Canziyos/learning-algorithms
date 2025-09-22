import numpy as np

def parse_data(path):
    """
    Parse TSPLIB-style .tsp file and return coordinates as a NumPy array.
    Shape: (n_cities, 2)
    """
    try:
        coords = []
        reading = False
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line == "NODE_COORD_SECTION":
                    reading = True
                    continue
                if line == "EOF":
                    break
                if reading:
                    _, x, y = line.split()
                    coords.append([float(x), float(y)])
        return np.array(coords, dtype=np.float64)
    except Exception as e:
        raise RuntimeError(f"Error while parsing {path}: {e}")

