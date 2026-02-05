import torch
import torch.nn as nn
from piq import ssim
from models.UNet import UNet
# from pytorch_msssim import ms_ssim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    model = UNet(3, 3, 3).to(DEVICE)
    state = torch.load(str("models/faceNet_best.pth"), map_location=DEVICE)
    model.load_state_dict(state)
    model.inc.double_conv[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.outc = torch.nn.Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1))
    return model


class FaceUNetLossLab(nn.Module):
    def __init__(self, lambda_ssim=0.3):
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_ab, target_ab):
        mse_part = self.mse_loss(pred_ab, target_ab)
        return mse_part
