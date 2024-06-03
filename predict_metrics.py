import logging
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

from sklearn.metrics import jaccard_score, f1_score

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
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

def calculate_metrics(pred_mask, true_mask, num_classes):
    """Calculate evaluation metrics."""
    pred_mask_flat = pred_mask.flatten()
    true_mask_flat = true_mask.flatten()

    iou = jaccard_score(true_mask_flat, pred_mask_flat, average='macro', labels=list(range(num_classes)))
    dice = f1_score(true_mask_flat, pred_mask_flat, average='macro', labels=list(range(num_classes)))
    accuracy = (pred_mask_flat == true_mask_flat).mean()

    return iou, dice, accuracy

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

def main(model_path, input_dir, mask_dir, output_dir, scale_factor=0.5, mask_threshold=0.5, bilinear=False, num_classes=2, n_channels=1, viz=False, no_save=False):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.png')))

    if len(in_files) != len(mask_files):
        logging.error('The number of input images and masks do not match.')
        return

    net = UNet(n_channels=n_channels, n_classes=num_classes, bilinear=bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {model_path}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(model_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    total_iou = 0
    total_dice = 0
    total_accuracy = 0
    num_images = len(in_files)

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)
        true_mask = np.array(Image.open(mask_files[i]))

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=scale_factor,
                           out_threshold=mask_threshold,
                           device=device)

        if not no_save:
            out_filename = os.path.join(output_dir, os.path.basename(filename))
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)

        # Calculate metrics
        iou, dice, accuracy = calculate_metrics(mask, true_mask, num_classes)
        total_iou += iou
        total_dice += dice
        total_accuracy += accuracy

        logging.info(f'Image {filename} - IoU: {iou:.4f}, Dice: {dice:.4f}, Accuracy: {accuracy:.4f}')

    avg_iou = total_iou / num_images
    avg_dice = total_dice / num_images
    avg_accuracy = total_accuracy / num_images

    logging.info(f'Average IoU: {avg_iou:.4f}, Average Dice: {avg_dice:.4f}, Average Accuracy: {avg_accuracy:.4f}')
