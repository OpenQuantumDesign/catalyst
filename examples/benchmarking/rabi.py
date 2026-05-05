import pennylane as qml
from catalyst.third_party.oqd import OQDDevice
import json

from oqd_core.interface.atomic import AtomicCircuit
from oqd_core.compiler.atomic.canonicalize import canonicalize_atomic_circuit_factory
from oqd_compiler_infrastructure import Chain, Post
from oqd_bare_metal.compiler.codegen import AtomicToBloodstoneV1
from oqd_bare_metal.compiler.codegen.builder import ARTIQPyBuilder
from oqd_bare_metal.compiler.optim import (
    SpectrumPrune,
    SpectrumUnwrapResets,
)
import ast_comments as ast

import pathlib
import numpy as np

from tempfile import TemporaryDirectory

from oqd_pipeline import OQD_PIPELINES

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


def generate_rabi(theta, pts, output_path=pathlib.Path("./expt")):

    # Setup Pennylane devices

    temp_dir = TemporaryDirectory()
    compile_results = pathlib.Path(temp_dir.name)
    openapl_file_name = "oqd_circuit_benchmarking.openapl.json"
    oqd_dev = OQDDevice(
        backend="default",
        wires=1,
        openapl_file_name=(compile_results / openapl_file_name).as_posix(),
    )

    dev = qml.device("default.qubit", wires=1)

    ########################################################################################

    for n in range(pts):
        print(
            f"\rCompiling realization {n + 1}/{pts}",
            end=" " * 32,
        )

        ########################################################################################

        # Pennylane Circuit

        # Simulate final state

        @qml.qnode(dev)
        def oqd_circuit_benchmarking_sim():
            qml.RX(n * theta / pts, wires=0)
            return qml.state()

        # Compile with Catalyst OQD pipelines

        @qml.qjit(pipelines=OQD_PIPELINES)
        @qml.set_shots(1)
        @qml.qnode(oqd_dev)
        def oqd_circuit_benchmarking():
            qml.RX(n * theta / pts, wires=0)
            return qml.counts(wires=0)

        ########################################################################################

        # Generate OpenAPL

        oqd_circuit_benchmarking()

        circuit = AtomicCircuit.model_validate_json(
            json.dumps(json.load(open(compile_results / openapl_file_name)), indent=2)
        )

        match circuit.protocol.sequence:
            case []:
                duration = 0
            case [parallel]:
                duration = parallel.sequence[0].duration
            case _:
                raise ValueError("Cannot extract duration")

        ########################################################################################

        # Compile AtomicCircuit to ARTIQ Python

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

        artiq_experiment.body.append(
            ast.Expr(
                ast.Comment(
                    value=f"# Final state = {oqd_circuit_benchmarking_sim()}",
                    inline=False,
                )
            )
        )

        save_duration_ast = [
            ARTIQPyBuilder.codegen_call(
                "self.set_dataset",
                keywords=[
                    ast.keyword("key", ast.Constant(value="duration")),
                    ast.keyword(
                        "value",
                        ARTIQPyBuilder.codegen_call(
                            "np.full",
                            args=[
                                ast.Constant(1),
                                ARTIQPyBuilder.codegen_attr("np.nan"),
                            ],
                            as_expr=False,
                        ),
                    ),
                ],
            ),
            ARTIQPyBuilder.codegen_call(
                "self.mutate_dataset",
                args=[
                    ast.Constant(value="duration"),
                    ast.Constant(value=0),
                    ast.Constant(value=duration),
                ],
            ),
        ]

        for line in reversed(save_duration_ast):
            artiq_experiment.body[2].body[1].body.insert(2, line)

        ########################################################################################

        expt_path = output_path / f"rabi/oqd_circuit_rabi_{n}.artiq.py"

        if not expt_path.parent.exists():
            expt_path.parent.mkdir()

        with open(expt_path, "w") as f:
            f.write(ast.unparse(ast.fix_missing_locations(artiq_experiment)))

    print()


########################################################################################

if __name__ == "__main__":
    # generate_rabi(5 * np.pi, 2)
    generate_rabi(5 * np.pi, 60)
