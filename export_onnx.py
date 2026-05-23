import torch
from model.u2net_loader import load_model

MODEL_PATH = "saved_models/u2netp/u2netp.pth"
ONNX_PATH = "u2netp_256.onnx"

def main():
    model, device = load_model(MODEL_PATH)
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["do", "d1", "d2", "d3", "d4", "d5", "d6"],
        opset_version=11,
    )

    print(f"ONNX export saved to: {ONNX_PATH}")

if __name__ == "__main__":
    main()



