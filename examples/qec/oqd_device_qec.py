from catalyst.debug.compiler_functions import get_compilation_stage
import pennylane as qml
from catalyst.third_party.oqd import OQDDevice

import os
import shutil
import pathlib
import numpy as np

from functools import partial

########################################################################################

for f in os.listdir():
    if f.startswith("oqd_circuit_qec") and os.path.isdir(f):
        shutil.rmtree(pathlib.Path(f))

compile_results = pathlib.Path("oqd_circuit_qec")
openapl_file_name = "oqd_circuit_qec.openapl.json"


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


oqd_dev = OQDDevice(
    backend="default",
    wires=26,
    openapl_file_name=(compile_results / openapl_file_name).as_posix(),
)


with open("rotated_surface_code.qasm", "r") as f:
    qasm_string = f.read()


@qml.set_shots(10)
@qml.qnode(oqd_dev)
def oqd_circuit_qec():
    qml.from_qasm(qasm_string)()
    return qml.counts(wires=0)


QJIT_CIRCUIT = qml.qjit(
    oqd_circuit_qec, pipelines=OQD_PIPELINES, keep_intermediate=True, verbose=True
)

print("{:=^100}".format("\033[1;32m Compiled circuit \033[0m"))
print(get_compilation_stage(QJIT_CIRCUIT, stage="IonDialectLoweringStage"))


########################################################################################

import json  # noqa: E402

print(qml.draw(oqd_circuit_qec)())

with open(compile_results / "oqd_circuit_qec.draw.txt", "w") as f:
    f.write(qml.draw(oqd_circuit_qec)())


QJIT_CIRCUIT()

print(json.dumps(json.load(open(compile_results / openapl_file_name)), indent=2))


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
        ),
    ),
)
optimization_pass = Chain(
    Post(SpectrumCoreRemapping()),
    Post(SpectrumUnwrapResets()),
    Post(SpectrumPrune()),
)

unopt_artiq_experiment = compiler(circuit)
artiq_experiment = optimization_pass(unopt_artiq_experiment)

print(ast.unparse(ast.fix_missing_locations(artiq_experiment)))

with open(compile_results / "oqd_circuit_qec.artiq.py", "w") as f:
    f.write(ast.unparse(ast.fix_missing_locations(artiq_experiment)))
