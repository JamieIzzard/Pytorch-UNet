import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from utils.data_loading import BasicDataset
from unet import UNet

def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

def load_model(model_path, n_channels=1, n_classes=1, bilinear=False, device='cpu'):
    net = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    net.to(device=device)
    state_dict = torch.load(model_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    return net, mask_values

def predict_and_save_masks(model_path, input_dir, output_dir, scale_factor=1, mask_threshold=0.5, bilinear=False, num_classes=1, n_channels=1, device='cpu'):
    net, mask_values = load_model(model_path, n_channels, num_classes, bilinear, device)
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            filepath = os.path.join(input_dir, filename)
            print(f'Predicting image {filepath} ...')
            img = Image.open(filepath)

            mask = predict_img(net=net,
                               full_img=img,
                               device=device,
                               scale_factor=scale_factor,
                               out_threshold=mask_threshold)

            out_filename = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_OUT.png")
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            print(f'Mask saved to {out_filename}')
