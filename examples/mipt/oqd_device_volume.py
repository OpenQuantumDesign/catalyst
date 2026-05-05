import os
import pathlib
import shutil

import numpy as np
import pennylane as qml
from catalyst.debug.compiler_functions import get_compilation_stage
from catalyst.third_party.oqd import OQDDevice

import pathlib
from datetime import datetime

import ast_comments as ast
import h5py
from oqd_compiler_infrastructure import Chain, Post
from oqd_dataschema import Dataset, Datastore, GroupBase

from oqd_core.compiler.atomic.canonicalize import canonicalize_atomic_circuit_factory
from oqd_core.interface.atomic import AtomicCircuit
from oqd_bare_metal.compiler.codegen import AtomicToTestbenchV2
from oqd_bare_metal.compiler.optim import (
    SpectrumCoreRemapping,
    SpectrumPrune,
    SpectrumUnwrapResets,
)
import scipy as sp

########################################################################################


class MeasurementsGroup(GroupBase):
    raw_photon_count: Dataset
    outcome: Dataset
    true_outcome: Dataset


class TrueStateGroup(GroupBase):
    state: Dataset


########################################################################################

for f in os.listdir():
    if f.startswith("oqd_circuit_volume") and os.path.isdir(f):
        shutil.rmtree(pathlib.Path(f))

compile_results = pathlib.Path("oqd_circuit_volume")
openapl_file_name = "oqd_circuit_volume.openapl.json"


toml_files = {
    "device-toml-loc": "/home/user/oqd-catalyst/scripts/calibration_data/device.toml",
    "qubit-toml-loc": "/home/user/oqd-catalyst/scripts/calibration_data/qubit.toml",
    "gate-to-pulse-toml-loc": "/home/user/oqd-catalyst/scripts/calibration_data/gate.toml",
}

toml_files = " ".join([f"{k}={v}" for k, v in toml_files.items()])


OQD_PIPELINES = [
    (
        "DeviceAgnosticPipeline",
        [
            "quantum-compilation-stage",
            "hlo-lowering-stage",
            "gradient-lowering-stage",
            "bufferization-stage",
        ],
    ),
    (
        "IonDecompositionStage",
        [
            "func.func(ions-decomposition)",
            "func.func(merge-rotations)",
            # "func.func(prune-zero-rotations)",
        ],
    ),
    (
        "IonDialectLoweringStage",
        [
            f"func.func(gates-to-pulses{{{toml_files}}})",
        ],
    ),
    ("IonToLLVMDialectConversion", ["convert-ion-to-llvm"]),
    ("MLIRToLLVMDialectConversion", ["llvm-dialect-lowering-stage"]),
]

########################################################################################


def block(num_qubits, block_rotation_record):
    for i in range(num_qubits // 2):
        qml.IsingXX(phi=np.pi / 2, wires=[2 * i, 2 * i + 1])

    for i in range(num_qubits):
        qml.Rot(block_rotation_record[i], np.pi / 2, -block_rotation_record[i], wires=i)

    for i in range(num_qubits // 2):
        qml.IsingXX(phi=np.pi / 2, wires=[2 * i + 1, (2 * i + 2) % num_qubits])

    for i in range(num_qubits):
        qml.Rot(block_rotation_record[i], np.pi / 2, -block_rotation_record[i], wires=i)


def measure_layer(num_qubits):
    """Mid-circuit measurement layer: measure qubit i if meas_record_t[i] is True."""
    out = []
    for i in range(num_qubits):
        out.append(qml.measure(wires=i))

    return out


########################################################################################

N = 10
T = 1
repetitions = 1000000

# Measurement rate.  Vary this to cross the MIPT phase transition.
p = 0.5

# Pre-generate a fixed (reproducible) measurement pattern: shape (T, N), dtype bool.
# np.random.seed(3141592653)


rotation_record = np.random.randint(0, 3, (T, N)) * np.pi / 4

########################################################################################


@qml.qnode(qml.device("default.qubit", wires=N))
def __oqd_circuit_volume():
    for i, rr in enumerate(rotation_record):
        block(N, rr)

    return qml.state()


true_state = __oqd_circuit_volume()
print(f"""
Rotation record: \033[1;32m{rotation_record}\033[0m
True state: \033[1;32m{true_state}\033[0m
""")

########################################################################################


@qml.set_shots(repetitions)
@qml.qnode(qml.device("default.qubit", wires=N))
def _oqd_circuit_volume():
    out = []
    for i, rr in enumerate(rotation_record):
        block(N, rr)
    out.extend(measure_layer(N))

    return qml.sample(out)


measurement_outcomes = _oqd_circuit_volume()
print(f"""
Rotation record: \033[1;32m{rotation_record}\033[0m
Measurement outcome: \033[1;32m{measurement_outcomes}\033[0m
""")

#######################################################################################

# oqd_dev = OQDDevice(
#     backend="default",
#     wires=N,
#     openapl_file_name=(compile_results / openapl_file_name).as_posix(),
# )


# @qml.set_shots(10)
# @qml.qnode(oqd_dev)
# def oqd_circuit_volume():
#     for i, rr in enumerate(rotation_record):
#         block(N, rr)
#     measure_layer(N)
#     return qml.counts(wires=0)


# print("{:=^100}".format("\033[1;32m Compiling circuit in Catalyst... \033[0m"))
# QJIT_CIRCUIT = qml.qjit(
#     oqd_circuit_volume, pipelines=OQD_PIPELINES, keep_intermediate=True, verbose=True
# )

# # print("{:=^100}".format("\033[1;32m Compiled circuit \033[0m"))
# # print(get_compilation_stage(QJIT_CIRCUIT, stage="IonDialectLoweringStage"))

# ########################################################################################

# import json  # noqa: E402

# # print(qml.draw(oqd_circuit_volume)())

# with open(compile_results / "oqd_circuit_volume.draw.txt", "w") as f:
#     f.write(qml.draw(oqd_circuit_volume)())


# QJIT_CIRCUIT()

# print(json.dumps(json.load(open(compile_results / openapl_file_name)), indent=2))

########################################################################################

import toml


def simulate_measurements(measure_rec, duration, device_params, threshold):
    if isinstance(device_params, str):
        os.path.exists(device_params)
        with open(device_params, "r") as f:
            device_params = toml.load(f)

    elif isinstance(device_params, dict):
        device_params = device_params

    else:
        raise TypeError("Invalid type for device_params")

    downstate_flip_prob = 1 - sp.stats.expon.cdf(
        threshold + 1,
        loc=0,
        scale=device_params["detection"]["downstate_exp_lambda_per_us"]
        * duration
        / 1e3,
    )
    upstate_flip_prob = sp.stats.poisson.cdf(
        threshold,
        device_params["detection"]["upstate_poisson_lambda_per_us"] * duration / 1e3,
    )

    print(
        f"downstate flip=\033[1;32m{downstate_flip_prob}\033[0m, upstate_flip=\033[1;32m{upstate_flip_prob}\033[0m"
    )

    raw_photon_count = (1 - measure_rec) * np.random.exponential(
        device_params["detection"]["downstate_exp_lambda_per_us"] * duration / 1e3,
        measure_rec.shape,
    ).astype(int) + (measure_rec) * np.random.poisson(
        device_params["detection"]["upstate_poisson_lambda_per_us"] * duration / 1e3,
        measure_rec.shape,
    ).astype(int)

    return raw_photon_count


# circuit = AtomicCircuit.model_validate_json(
#     json.dumps(json.load(open(compile_results / openapl_file_name)), indent=2)
# )

# canonicalize = canonicalize_atomic_circuit_factory()
# circuit = canonicalize(circuit)


# raw_photon_counts = np.empty((repetitions, 10))
# for r in range(repetitions):
#     print(f"\rCompiling repetition {r + 1}/{repetitions}", end=" " * 80)

#     lowering = AtomicToTestbenchV2(
#         device_params="./calibration_data/testbench_params.toml",
#         measurement_record=measurement_outcomes[r],
#         verbose=False,
#     )

#     compiler2 = Chain(
#         Post(lowering),
#         Post(SpectrumCoreRemapping()),
#         Post(SpectrumUnwrapResets()),
#         Post(SpectrumPrune()),
#     )

#     artiq_experiment = compiler2(circuit)

#     raw_photon_counts[r] = np.array(
#         [m.photon_count for m in lowering.measurement_record]
#     )
# print()


threshold = 5

raw_photon_counts = simulate_measurements(
    measurement_outcomes,
    1e4,
    pathlib.Path("./calibration_data/testbench_params.toml").resolve().as_posix(),
    threshold,
)

datafile = pathlib.Path("oqd_circuit_volume/oqd_circuit_volume.result.h5")

store = Datastore.model_validate_hdf5(datafile) if datafile.exists() else Datastore()
raw_photon_count_ds = Dataset(data=raw_photon_counts)
outcome_ds = Dataset(data=(raw_photon_counts > threshold).astype(int))
true_outcome_ds = Dataset(data=(measurement_outcomes))
store.add(
    measurements=MeasurementsGroup(
        raw_photon_count=raw_photon_count_ds,
        outcome=outcome_ds,
        true_outcome=true_outcome_ds,
        attrs={
            "created_at": datetime.now().isoformat(),
            "threshold": threshold,
        },
    )
)
true_state_ds = Dataset(data=true_state)
store.add(
    true_state=TrueStateGroup(
        state=true_state_ds,
        attrs={
            "created_at": datetime.now().isoformat(),
        },
    )
)
store.model_dump_hdf5(datafile, "a")
