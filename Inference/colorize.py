import PIL
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lab_to_rgb(L_tensor, ab_tensor):
    if L_tensor.ndim == 3:
        L = L_tensor.squeeze(0).cpu().numpy()
    else:
        L = L_tensor.cpu().numpy()

    ab = ab_tensor.cpu().numpy().transpose(1,2,0)

    L_cv = (L * 255.0).clip(0,255).astype(np.uint8)
    ab_cv = (ab * 128.0 + 128.0).clip(0,255).astype(np.uint8)

    lab_cv = np.zeros((L_cv.shape[0], L_cv.shape[1], 3), dtype=np.uint8)
    lab_cv[..., 0] = L_cv
    lab_cv[..., 1] = ab_cv[..., 0]
    lab_cv[..., 2] = ab_cv[..., 1]

    rgb = cv2.cvtColor(lab_cv, cv2.COLOR_LAB2RGB)  
    return rgb



def colorize(img: PIL.Image, model, device=DEVICE, save_path="result.png"):
    # get the width and height of the input image
    w, h = img.size
    
    model.eval()
    img_np = np.array(img)
    img_np = cv2.resize(img_np, (256, 256))

    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)

    L = lab[..., 0] / 255.0
    L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        ab_output = model(L_tensor)
    output_rgb = lab_to_rgb(L_tensor[0], ab_output[0])
    # resize the output to the original image size
    output_rgb = cv2.resize(output_rgb, (w, h))

    plt.imsave(save_path, output_rgb)
    return output_rgb
