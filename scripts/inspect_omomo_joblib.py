import joblib
import numpy as np

TRAIN_JOBLIB_PATH = "/home/learning/Documents/g1-gmr/OMOMO_DATA/OMOMO_p_files/train_diffusion_manip_seq_joints24.p"


def describe_array(name, value, max_slice=3):
    if hasattr(value, "shape"):
        print(f"  {name}: type={type(value)}, shape={value.shape}")
        squeeze_slice = value
        if value.ndim == 1:
            squeeze_slice = value[:max_slice]
        elif value.ndim >= 2:
            slicer = [slice(None)] * value.ndim
            slicer[0] = slice(0, min(max_slice, value.shape[0]))
            squeeze_slice = value[tuple(slicer)]
        print(f"    sample: {np.array(squeeze_slice)}")
    else:
        print(f"  {name}: type={type(value)}, value={value}")


def main():
    print(f"Loading joblib from {TRAIN_JOBLIB_PATH}")
    motion_data = joblib.load(TRAIN_JOBLIB_PATH)
    print(f"Total sequences: {len(motion_data)}")

    first_key = list(motion_data.keys())[0]
    print(f"First key: {first_key}")
    first_value = motion_data[first_key]
    print(f"Type(first_value): {type(first_value)}")

    if isinstance(first_value, dict):
        print("Top-level keys:", list(first_value.keys()))
        for key, value in first_value.items():
            key_lower = key.lower()
            looks_like_object = any(part in key_lower for part in ["obj", "object", "peg", "hole", "cube", "box"])
            if looks_like_object:
                describe_array(key, value)
            else:
                if hasattr(value, "shape"):
                    print(f"{key}: shape={value.shape}, dtype={getattr(value, 'dtype', 'unknown')}")
                else:
                    print(f"{key}: type={type(value)}")

        object_keys = [k for k in first_value.keys() if any(part in k.lower() for part in ["obj", "object", "peg", "hole", "cube", "box"])]
        print("Potential object-related keys:", object_keys)
    else:
        print("First value is not a dict; unable to list internal keys.")


if __name__ == "__main__":
    main()
