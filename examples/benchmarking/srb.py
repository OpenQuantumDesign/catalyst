from qiskit_experiments.library import StandardRB
from qiskit import qasm2
from qiskit import transpile
import pennylane as qml
from catalyst.third_party.oqd import OQDDevice
import json

from oqd_core.interface.atomic import AtomicCircuit
from oqd_core.compiler.atomic.canonicalize import canonicalize_atomic_circuit_factory
from oqd_compiler_infrastructure import Chain, Post
from oqd_bare_metal.compiler.codegen import AtomicToBloodstoneV1
from oqd_bare_metal.compiler.optim import (
    SpectrumPrune,
    SpectrumUnwrapResets,
)
import ast_comments as ast

import pathlib

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


def generate_circuit(
    length, num_samples, initial_state=0, output_path=pathlib.Path("./expt")
):
    if not output_path.exists():
        output_path.mkdir()

    qubits = (0,)
    lengths = [length]
    num_samples = num_samples

    # Qiskit experiments stardard randomized benchmarking

    rb = StandardRB(qubits, lengths, num_samples=num_samples)
    circuits = [
        transpile(
            circ,
            basis_gates=["s", "sxdg", "h", "z", "x", "y"],
            optimization_level=0,
        )
        for circ in rb.circuits()
    ]

    ########################################################################################

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

    for n, circ in enumerate(circuits):
        print(
            f"\rCompiling realization {n + 1}/{len(circuits)} [{length=}, {initial_state=}]",
            end=" " * 32,
        )

        ########################################################################################

        # Pennylane Circuit

        # Simulate final state

        @qml.qnode(dev)
        def oqd_circuit_benchmarking_sim():
            qml.X(wires=0) if initial_state else None
            qml.from_qasm(qasm2.dumps(remove_unsupported(circ)))()
            return qml.state()

        # Compile with Catalyst OQD pipelines

        @qml.qjit(pipelines=OQD_PIPELINES)
        @qml.set_shots(1)
        @qml.qnode(oqd_dev)
        def oqd_circuit_benchmarking():
            qml.X(wires=0) if initial_state else None
            qml.from_qasm(qasm2.dumps(circ))()
            return qml.counts(wires=0)

        ########################################################################################

        # Generate OpenAPL

        oqd_circuit_benchmarking()

        circuit = AtomicCircuit.model_validate_json(
            json.dumps(json.load(open(compile_results / openapl_file_name)), indent=2)
        )

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

        ########################################################################################

        expt_path = (
            output_path
            / f"T={length}-State={initial_state}/oqd_circuit_benchmarking_{n}.artiq.py"
        )

        if not expt_path.parent.exists():
            expt_path.parent.mkdir()

        with open(expt_path, "w") as f:
            f.write(ast.unparse(ast.fix_missing_locations(artiq_experiment)))

    print()


########################################################################################

if __name__ == "__main__":
    # generate_circuit(5, 50, 0)
    # generate_circuit(5, 50, 1)

    generate_circuit(10, 50, 0)
    generate_circuit(10, 50, 1)

    # generate_circuit(20, 50, 0)
    # generate_circuit(20, 50, 1)

    # generate_circuit(1, 1, 0)
    # generate_circuit(1, 1, 1)
