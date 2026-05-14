import lmdb
from pathlib import Path

import numpy as np

from fractions import Fraction
from collections import OrderedDict
import toml

try:
    import pybase64 as base64
except ImportError:
    import base64

from datetime import datetime

########################################################################################


def _nparray(shape, dtype, data):
    a = np.frombuffer(base64.b64decode(data), dtype=dtype)
    a = a.copy()
    return a.reshape(shape)


def _npscalar(ty, data):
    return np.frombuffer(base64.b64decode(data), dtype=ty)[0]


_eval_dict = {
    "__builtins__": {},
    "null": None,
    "false": False,
    "true": True,
    "inf": np.inf,
    "slice": slice,
    "nan": np.nan,
    "Fraction": Fraction,
    "OrderedDict": OrderedDict,
    "nparray": _nparray,
    "npscalar": _npscalar,
}


def decode(s):
    """
    Parses a string in the Python syntax, reconstructs the corresponding
    object, and returns it.
    **Shouldn't** be used with untrusted inputs, as it can cause vulnerability against injection attacks.
    """
    return eval(s, _eval_dict, {})


########################################################################################


path = Path("./dataset_db.mdb")
bloodstone_toml = Path("./bloodstone_params.toml")
# Your existing code to load the data
records = []
lmdb_file = lmdb.open(path.as_posix(), subdir=False, map_size=2**30)
data = dict()
with lmdb_file.begin() as txn:
    for key, value_and_metadata in txn.cursor():
        value, metadata = decode(value_and_metadata.decode())
        data[key.decode()] = (value, metadata)


with open(bloodstone_toml, "r") as f:
    bloodstone_dict = toml.load(f)

bloodstone_dict["doppler_cooling"]["frequency"] = data["Cooling_freq_param"][0] / 1e3
bloodstone_dict["doppler_cooling"]["phase"] = data["Cooling_phase_param"][0]
bloodstone_dict["doppler_cooling"]["attenuation"] = data["Cooling_atten_param"][0]
bloodstone_dict["doppler_cooling"]["sideband_frequency"] = (
    data["Cooling_sideband_freq_param"][0] / 1e3
)
bloodstone_dict["doppler_cooling"]["sideband_phase"] = data[
    "Cooling_sideband_phase_param"
][0]
bloodstone_dict["doppler_cooling"]["sideband_attenuation"] = data[
    "Cooling_sideband_atten_param"
][0]

bloodstone_dict["doppler_cooling"]["duration"] = int(
    data["Cooling_time_param"][0] * 1e9
)

bloodstone_dict["optical_pumping"]["frequency"] = data["Pumping_freq_param"][0] / 1e3
bloodstone_dict["optical_pumping"]["phase"] = data["Pumping_phase_param"][0]
bloodstone_dict["optical_pumping"]["attenuation"] = data["Pumping_atten_param"][0]
bloodstone_dict["optical_pumping"]["duration"] = int(
    data["Pumping_time_param"][0] * 1e9
)


bloodstone_dict["detection"]["frequency"] = data["Detection_freq_param"][0] / 1e3
bloodstone_dict["detection"]["phase"] = data["Detection_phase_param"][0]
bloodstone_dict["detection"]["attenuation"] = data["Detection_atten_param"][0]
bloodstone_dict["detection"]["duration"] = int(data["Detection_time_param"][0] * 1e9)

bloodstone_dict["raman1"]["raw_frequency"] = 5306062750305622.0 - (
    2 * np.pi * data["Raman_carrier_freq_param"][0] / 1e3
)


if not (bloodstone_toml.parent / "archive").exists():
    (bloodstone_toml.parent / "archive").mkdir()

bloodstone_toml.rename(
    bloodstone_toml.parent
    / "archive"
    / f"{bloodstone_toml.stem}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.toml"
)


with open(bloodstone_toml, "w") as f:
    toml.dump(bloodstone_dict, f)


########################################################################################

pi_time = data["Raman_carrier_pi_time_param"][0]


rabi = np.pi / pi_time

raw_rabi = np.sqrt((2 * 208570336271826.38) * rabi)


print(
    f"Single qubit gate Rabi = \033[1;32m{raw_rabi}\033[0m, pi time = \033[1;32m{pi_time}\033[0m"
)
