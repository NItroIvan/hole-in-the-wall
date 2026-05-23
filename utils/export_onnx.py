import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from model.u2net_loader import load_model


MODEL_PATH = ROOT_DIR / "saved_models" / "u2netp" / "u2netp.pth"
ONNX_PATH = ROOT_DIR / "onnx_models" / "u2netp_256.onnx"


def main():
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    model, device = load_model(MODEL_PATH)
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["d0", "d1", "d2", "d3", "d4", "d5", "d6"],
        opset_version=11,
        dynamo=False,
    )

    print(f"ONNX export saved to: {ONNX_PATH}")


if __name__ == "__main__":
    main()