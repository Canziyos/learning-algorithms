# --- put these helpers near the top of your script ---
import numpy as np
from dense import Dense

def save_model_npz(path, layers, x_mu, x_std, y_mu, y_std, splits=None):
    payload = {
        "x_mu": x_mu, "x_std": x_std,
        "y_mu": y_mu, "y_std": y_std,
        "ins": np.array([L.input_s for L in layers]),
        "outs": np.array([L.output_s for L in layers]),
        "acts": np.array([getattr(L.activation, "__name__", "identity") for L in layers], dtype=object),
        "dtype": np.array(str(layers[0].w.dtype)),
    }
    # parameters
    for i, L in enumerate(layers):
        payload[f"w{i}"] = L.w
        payload[f"b{i}"] = L.b
    # optional: keep the exact splits for reproducibility
    if splits is not None:
        it, iv, ie = splits
        payload["it"] = it
        payload["iv"] = iv
        payload["ie"] = ie
    np.savez(path, **payload)

def load_model_npz(path):
    Z = np.load(path, allow_pickle=True)
    ins  = Z["ins"].astype(int)
    outs = Z["outs"].astype(int)
    acts = [str(a) for a in Z["acts"]]
    dtype = np.float64 if str(Z["dtype"]) == "float64" else np.float32

    layers = []
    for i, (inp, out, act) in enumerate(zip(ins, outs, acts)):
        L = Dense(input_s=inp, output_s=out, activation=act, init="xavier", dtype=dtype)
        L.w[...] = Z[f"w{i}"]
        L.b[...] = Z[f"b{i}"]
        layers.append(L)

    x_mu, x_std = Z["x_mu"], Z["x_std"]
    y_mu, y_std = Z["y_mu"], Z["y_std"]

    splits = None
    if all(k in Z for k in ("it", "iv", "ie")):
        splits = (Z["it"], Z["iv"], Z["ie"])

    return layers, (x_mu, x_std, y_mu, y_std), splits
