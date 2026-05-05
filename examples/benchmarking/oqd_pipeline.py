toml_files = {
    "device-toml-loc": "/home/user/oqd-catalyst/examples/calibration_data/device.toml",
    "qubit-toml-loc": "/home/user/oqd-catalyst/examples/calibration_data/qubit.toml",
    "gate-to-pulse-toml-loc": "/home/user/oqd-catalyst/examples/calibration_data/gate.toml",
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
