import os
import pathlib
import shutil

import numpy as np
import pennylane as qml
from catalyst.debug.compiler_functions import get_compilation_stage
from catalyst.third_party.oqd import OQDDevice

########################################################################################

for f in os.listdir():
    if f.startswith("oqd_circuit_mipt") and os.path.isdir(f):
        shutil.rmtree(pathlib.Path(f))

compile_results = pathlib.Path("oqd_circuit_mipt")
openapl_file_name = "oqd_circuit_mipt.openapl.json"


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


def measure_layer(num_qubits, meas_record_t):
    """Mid-circuit measurement layer: measure qubit i if meas_record_t[i] is True."""
    out = []
    for i in range(num_qubits):
        if meas_record_t[i]:
            out.append(qml.measure(wires=i))

    return out


########################################################################################

N = 10
T = 2 * N

# Measurement rate.  Vary this to cross the MIPT phase transition.
p = 0.5

# Pre-generate a fixed (reproducible) measurement pattern: shape (T, N), dtype bool.
# np.random.seed(3141592653)
measurement_record = np.random.random((T, N)) < p

print(f"No. of measurements: \033[1;32m{measurement_record.sum()}\033[0m")

measurement_record = measurement_record.tolist()
rotation_record = np.random.randint(0, 3, (T, N)) * np.pi / 4


@qml.set_shots(1)
@qml.qnode(qml.device("default.qubit", wires=N))
def _oqd_circuit_mipt():
    out = []
    for i, rr in enumerate(rotation_record):
        block(N, rr)
        out.extend(measure_layer(N, measurement_record[i]))

    return qml.sample(out)


measurement_outcomes = _oqd_circuit_mipt()[0]
print(f"""
Rotation record: \033[1;32m{rotation_record}\033[0m
Measurement record: \033[1;32m{measurement_record}\033[0m
Measurement outcome: \033[1;32m{measurement_outcomes}\033[0m
""")

#######################################################################################

oqd_dev = OQDDevice(
    backend="default",
    wires=N,
    openapl_file_name=(compile_results / openapl_file_name).as_posix(),
)


@qml.set_shots(10)
@qml.qnode(oqd_dev)
def oqd_circuit_mipt():
    for i, rr in enumerate(rotation_record):
        block(N, rr)
        measure_layer(N, measurement_record[i])
    return qml.counts(wires=0)


print("{:=^100}".format("\033[1;32m Compiling circuit in Catalyst... \033[0m"))
QJIT_CIRCUIT = qml.qjit(
    oqd_circuit_mipt, pipelines=OQD_PIPELINES, keep_intermediate=True, verbose=True
)

# print("{:=^100}".format("\033[1;32m Compiled circuit \033[0m"))
# print(get_compilation_stage(QJIT_CIRCUIT, stage="IonDialectLoweringStage"))

########################################################################################

import json  # noqa: E402

# print(qml.draw(oqd_circuit_mipt)())

with open(compile_results / "oqd_circuit_mipt.draw.txt", "w") as f:
    f.write(qml.draw(oqd_circuit_mipt)())


QJIT_CIRCUIT()

# print(json.dumps(json.load(open(compile_results / openapl_file_name)), indent=2))

########################################################################################


import ast_comments as ast  # noqa: E402
from oqd_compiler_infrastructure import Chain, Post  # noqa: E402
from oqd_core.compiler.atomic.canonicalize import (  # noqa: E402
    canonicalize_atomic_circuit_factory,  # noqa: E402
)
from oqd_core.interface.atomic import AtomicCircuit  # noqa: E402

from oqd_bare_metal.compiler.codegen import AtomicToTestbenchV2  # noqa: E402
from oqd_bare_metal.compiler.optim import (  # noqa: E402
    SpectrumCoreRemapping,
    SpectrumPrune,
    SpectrumUnwrapResets,
)

circuit = AtomicCircuit.model_validate_json(
    json.dumps(json.load(open(compile_results / openapl_file_name)), indent=2)
)


compiler = Chain(
    canonicalize_atomic_circuit_factory(),
    Post(
        AtomicToTestbenchV2(
            device_params="./calibration_data/testbench_params.toml",
            measurement_record=measurement_outcomes,
        ),
    ),
)
optimization_pass = Chain(
    Post(SpectrumCoreRemapping()),
    Post(SpectrumUnwrapResets()),
    Post(SpectrumPrune()),
)


print("{:=^100}".format("\033[1;32m Compiling circuit in OQD... \033[0m"))
unopt_artiq_experiment = compiler(circuit)
artiq_experiment = optimization_pass(unopt_artiq_experiment)

# print(ast.unparse(ast.fix_missing_locations(artiq_experiment)))

with open(compile_results / "oqd_circuit_mipt.artiq.py", "w") as f:
    f.write(ast.unparse(ast.fix_missing_locations(artiq_experiment)))
