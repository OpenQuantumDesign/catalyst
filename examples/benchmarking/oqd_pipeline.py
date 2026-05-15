import pathlib


device_toml_loc = pathlib.Path("../calibration_data/device.toml").resolve().as_posix()
qubit_toml_loc = pathlib.Path("../calibration_data/qubit.toml").resolve().as_posix()
gate_to_pulse_toml_loc = (
    pathlib.Path("../calibration_data/gate.toml").resolve().as_posix()
)

toml_files = {
    "device-toml-loc": device_toml_loc,
    "qubit-toml-loc": qubit_toml_loc,
    "gate-to-pulse-toml-loc": gate_to_pulse_toml_loc,
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
