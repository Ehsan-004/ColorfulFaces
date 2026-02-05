import torch
from .UNet import UNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(state_path = "Inference/weights/base.pth", device=DEVICE):
    model = UNet(3, 3, depth=3).to(device)
    model.inc.double_conv[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.outc = torch.nn.Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1))
    state = torch.load(str(state_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
