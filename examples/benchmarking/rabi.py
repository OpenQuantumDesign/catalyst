from qiskit_experiments.library import StandardRB
from qiskit import qasm2
from qiskit import transpile
from catalyst.debug.compiler_functions import get_compilation_stage
import pennylane as qml
from catalyst.third_party.oqd import OQDDevice
import json

from oqd_core.interface.atomic import AtomicCircuit
from oqd_core.compiler.atomic.canonicalize import canonicalize_atomic_circuit_factory
from oqd_compiler_infrastructure import Chain, Post
from oqd_bare_metal.compiler.codegen import AtomicToBloodstoneV1
from oqd_bare_metal.compiler.optim import (
    SpectrumCoreRemapping,
    SpectrumPrune,
    SpectrumUnwrapResets,
)
import ast_comments as ast

import os
import shutil
import pathlib
import numpy as np

from functools import partial

########################################################################################


def remove_unsupported(circuit):
    circuit = circuit.copy()
    delete_list = []
    for n, line in enumerate(circuit.data):
        if line.operation.name in ["measure", "barrier"]:
            delete_list.append(n)

    for n in reversed(delete_list):
        del circuit.data[n]

    return circuit


########################################################################################


def generate_rabi(pts, output_path=pathlib.Path("./expt")):

    ########################################################################################

    compile_results = pathlib.Path("temp")
    if not compile_results.exists():
        compile_results.mkdir()
    openapl_file_name = "oqd_circuit_benchmarking.openapl.json"

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
                "func.func(prune-zero-rotations)",
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
        wires=1,
        openapl_file_name=(compile_results / openapl_file_name).as_posix(),
    )

    dev = qml.device("default.qubit", wires=1)

    for n in range(pts):
        print(
            f"\rCompiling realization {n + 1}/{pts}",
            end=" " * 32,
        )

        @qml.qjit(pipelines=OQD_PIPELINES)
        @qml.set_shots(1)
        @qml.qnode(oqd_dev)
        def oqd_circuit_benchmarking():
            qml.RX(n * 5 * np.pi / pts, wires=0)
            return qml.counts(wires=0)

        oqd_circuit_benchmarking()

        ########################################################################################

        circuit = AtomicCircuit.model_validate_json(
            json.dumps(json.load(open(compile_results / openapl_file_name)), indent=2)
        )

        compiler = Chain(
            canonicalize_atomic_circuit_factory(),
            Post(
                AtomicToBloodstoneV1(
                    device_params="../calibration_data/bloodstone_params.toml",
                ),
            ),
        )
        optimization_pass = Chain(
            Post(SpectrumUnwrapResets(cores={20})),
            Post(SpectrumPrune()),
        )

        unopt_artiq_experiment = compiler(circuit)
        artiq_experiment = optimization_pass(unopt_artiq_experiment)

        artiq_experiment.body.extend(
            [
                ast.Expr(value=ast.Comment(value=f"# {line}", inline=False))
                if i == 0
                else ast.Comment(value=f"# {line}", inline=False)
                for i, line in enumerate(
                    qml.draw(oqd_circuit_benchmarking)().splitlines()
                )
                if line
            ]
        )

        ########################################################################################

        @qml.qnode(dev)
        def oqd_circuit_benchmarking_sim():
            qml.RX(n * 5 * np.pi / pts, wires=0)
            return qml.state()

        artiq_experiment.body.append(
            ast.Expr(
                ast.Comment(
                    value=f"# Final state = {oqd_circuit_benchmarking_sim()}",
                    inline=False,
                )
            )
        )

        ########################################################################################

        expt_path = output_path / f"rabi/oqd_circuit_benchmarking_{n}.artiq.py"

        if not expt_path.parent.exists():
            expt_path.parent.mkdir()

        with open(expt_path, "w") as f:
            f.write(ast.unparse(ast.fix_missing_locations(artiq_experiment)))

    print()


########################################################################################

if __name__ == "__main__":
    generate_rabi(60)
