import torch
from model import U2NETP

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = U2NETP(3, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device