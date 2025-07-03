import torch
from cellpose import io, models
from cellpose.io import imread
from cellpose import plot
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from tabulate import tabulate
from PIL import Image
from skimage.color import label2rgb
from model import build_model
from preprocessing import binarizeMaskDir, cropLineBelow, slidingWindowPatchDir, slidingWindowPatch
from numpy import save
from dataset import BiofilmDataset, TestDataset, splitDatasetInDirs, leaveOneSEMimageOut
import matplotlib.pyplot as plt
import re
from skimage import measure
pattern = r'\.(\d+)_(\d+)\.png$'

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    jaccard_score
)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_checkpoint(model, checkpoint_path):
    model = nn.DataParallel(model) 
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.module    

    print(f"Model loaded from: {checkpoint_path}")

def iou_score(preds, masks, eps=1e-6): 
    intersection = torch.logical_and(preds == 1, masks == 1).sum(dim=(1, 2))
    union = torch.logical_or(preds == 1, masks == 1).sum(dim=(1, 2))
    iou = intersection / (union + eps)
    return iou

def dice_score(preds, masks, eps=1e-6):
    intersection = torch.logical_and(preds == 1, masks == 1).sum(dim=(1, 2))
    total = preds.sum(dim=(1, 2)) + masks.sum(dim=(1, 2))
    dice = (2 * intersection) / (total + eps)
    return dice

def fmeasure_score(preds, masks, positiveClass = 1, negativeClass = 0, eps=1e-6):
    TP = torch.logical_and(preds == positiveClass, masks == positiveClass).sum(dim=(1, 2))
    FP = torch.logical_and(preds == positiveClass, masks == negativeClass).sum(dim=(1, 2))
    TN = torch.logical_and(preds == negativeClass, masks == negativeClass).sum(dim=(1, 2))
    FN = torch.logical_and(preds == negativeClass, masks == positiveClass).sum(dim=(1, 2))
    Fmeasure = (2 * TP) / (2*TP + FP + FN + eps)
    return Fmeasure

def p4measure_score(preds, masks, eps=1e-6):
    TP = torch.logical_and(preds == 1, masks == 1).sum(dim=(1, 2))
    FP = torch.logical_and(preds == 1, masks == 0).sum(dim=(1, 2))
    TN = torch.logical_and(preds == 0, masks == 0).sum(dim=(1, 2))
    FN = torch.logical_and(preds == 0, masks == 1).sum(dim=(1, 2))
    invTP = 1.0 / (TP + eps)
    invTN = 1.0 / (TN + eps)
    p4measure = 4.0 / ((invTP + invTN) * (FP + FN) + 4)
    return p4measure


def p4multicalss(matr):
    ps = []
    for i in range(matr.shape[0]):
        tp = matr[i, i]
        fn = np.sum(matr[:, i]) - tp
        fp = np.sum(matr[i, :]) - tp
        tn = np.sum(matr.ravel()) - (tp + fn + fp+1e-6)
        ps.append(4 / ((1 / (tp+1e-6) + 1 / (tn+1e-6)) * (fp + fn) + 4))
    den = np.sum([1 / p for p in ps])
    return matr.shape[0] / den


def filter_masks_by_area(masks, probs, area_factor=3.0):
    props = measure.regionprops(masks)
    areas = [prop.area for prop in props]
    avg_area = np.mean(areas)
    area_threshold = avg_area * area_factor
    filtered_probs = np.copy(probs)
    filtered_masks = np.copy(masks)

    for prop in props:
        if prop.area > area_threshold:
            filtered_masks[filtered_masks == prop.label] = 0
            filtered_probs[filtered_masks == prop.label] = 0.0

    return filtered_masks, filtered_probs

def filter_masks_by_area_and_shape(masks, probs, area_factor=3, min_circularity=0.75, reference_area=1140, reference_circularity = 0.946):
    props = measure.regionprops(masks)
    areas = [prop.area for prop in props]
    avg_area = reference_area
    area_threshold = avg_area * area_factor

    filtered_probs = np.copy(probs)
    filtered_masks = np.copy(masks)
    for prop in props:
        ecc = prop.eccentricity  
        
        if (prop.area > area_threshold) or (prop.eccentricity < min_circularity) or (prop.area < (avg_area / 2)):
            filtered_masks[filtered_masks == prop.label] = 0
            filtered_probs[filtered_masks == prop.label] = 0.0

    return filtered_masks, filtered_probs

def testOneSEMimageUnetCellpose(imgDir = None, coloredMaskDir = None, 
                                biofilmMaskDir = None, bacteriaMaskDir = None,
                       outputDir = None, 
                       configPath = None, checkpointPath = None, targetRGB=(36, 179, 83)):
    
    imgProcessedDir = slidingWindowPatchDir(imgDir = imgDir, 
                       imgMode= "RGB",
                       patch_size = (512, 512),
                       stride = (128, 128),
                       save_dir = outputDir + r"/images",
                       visualize=False
    )
    maskProcessedDir = slidingWindowPatchDir(imgDir = biofilmMaskDir, 
                       imgMode= "L",
                       patch_size = (512, 512),
                       stride = (128, 128),
                       save_dir = outputDir + r"/masks",
                       visualize=False
    )
    coloredMasksProcessedDir = slidingWindowPatchDir(imgDir = coloredMaskDir, 
                       imgMode= "RGB",
                       patch_size = (512, 512),
                       stride = (128, 128),
                       save_dir = outputDir + r"/colored-masks",
                       visualize=False
    )

    with open(configPath) as f:
        cfg = json.load(f)
        
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"---> DEVICE IS: {device}")
    test_dataset = BiofilmDataset(
        image_dir=imgProcessedDir,
        mask_dir=maskProcessedDir,
        colored_mask_dir=coloredMasksProcessedDir,
        mode="test"
    )
    test_loader = DataLoader(test_dataset,
                             batch_size=2,
                             shuffle=False)

    model = build_model(cfg["model"]).to(device)   
    load_checkpoint(model, checkpointPath)
    model.eval()
    
    height, width = 1792, 2560 
    probsCount = np.zeros((height, width), dtype=float)
    biofilmProbs = np.zeros((height, width), dtype=float)
    biofilmPredictions = np.zeros((height, width), dtype=float)

    imgName = os.path.splitext(os.path.basename(os.listdir(imgDir)[0]))[0]
    imgPath = os.path.join(imgDir, os.listdir(imgDir)[0])
    maskPath = os.path.join(biofilmMaskDir, os.listdir(biofilmMaskDir)[0])
    coloredMaskPath = os.path.join(coloredMaskDir, os.listdir(coloredMaskDir)[0])
    
    probsCount = np.load("/home/VizaVi/unetpp-torch/config/probsCount.npy")
    metrics_text = ""
    
    # ==== UNET++ PROCESSING ====
    with torch.no_grad():
        for i, (images, imgpaths, masks, maskpaths, color_masks, cmaskspaths) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            outputs = outputs.cpu()
                   
            for idx in range(images.size(0)):
                img_path = imgpaths[idx]
                img_name = os.path.basename(img_path)
                
                match = re.search(pattern, img_name)
                if match:
                    x = int(match.group(1))
                    y = int(match.group(2))
                
                output_np = outputs[idx].numpy()[1]
                
                biofilmProbs[y:y+512, x:x+512] += output_np
                #probsCount[y:y+512, x:x+512] += 1
                #print(f'---> {img_name} <---')

    
    threshold = 0.5
    biofilmProbs = biofilmProbs / probsCount 
    # np.save("/home/VizaVi/unetpp-torch/config/probsCount.npy", probsCount)
    biofilmPredictions = (biofilmProbs > threshold).astype(np.uint8) 
    
    origImg = Image.open(imgPath).convert("RGB")
    origImg = cropLineBelow(origImg, countPx=128)
    origImgNP = np.array(origImg) 
    cleaned_image = origImgNP.copy()
    cleaned_image[biofilmPredictions == 1] = 0 #black 
 
    # ==== CELLPOSE PROCESSING ====
    
    model_cp = models.CellposeModel(gpu=False)
    singlePredictions, flows, styles = model_cp.eval(cleaned_image, channels=[0, 0])
    singleProbs = sigmoid(flows[2])
    singlePredictions, singleProbs = filter_masks_by_area_and_shape(singlePredictions, singleProbs)
    
    # GT MATRIX
    mask = Image.open(maskPath).convert("L")
    mask = cropLineBelow(mask, countPx=128)
    biofilmGT = (np.array(mask) == 255).astype(np.uint8)
    
    bacteriaMask = Image.open(bacteriaMaskDir).convert("L")
    bacteriaMask = cropLineBelow(bacteriaMask, countPx=128)
    singleGT = (np.array(bacteriaMask) == 255).astype(np.uint8)
    
    # PREDS MATRIX
    singlePredictions = np.array(singlePredictions != 0, dtype=np.uint8)
    
    backgroundGT = np.zeros((height, width), dtype=np.uint8)
    backgroundPreds = np.zeros((height, width), dtype=np.uint8)
    backgroundGT[(singleGT == 0) & (biofilmGT == 0)] = 1
    backgroundPreds[(singlePredictions == 0) & (biofilmPredictions == 0)] = 1
    
        ## PROCESSING OVERLAP
    biofilm_mask = (biofilmPredictions == 1)
    bacteria_mask = (singlePredictions == 1)
    overlap = (biofilmPredictions == 1) & (singlePredictions == 1)

        ## DECISION BY PROBS
    prefer_biofilm = biofilmProbs > singleProbs   
    biofilm_mask[overlap] = prefer_biofilm[overlap]
    biofilmPredictions[overlap] = prefer_biofilm[overlap]

    prefer_bacteria = singleProbs >= biofilmProbs
    bacteria_mask[overlap] = prefer_bacteria[overlap]
    singlePredictions[overlap] = prefer_bacteria[overlap]
    
    # SETTINGS FOR VIZUALIZATION
    biofilm_color = np.array([36, 179, 83, 255], dtype=np.uint8)  # RGBA
    bacteria_color = np.array([184, 61, 245, 255], dtype=np.uint8)
    difference_color = np.array([255, 125, 0, 255], dtype=np.uint8)
    
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    overlay[biofilm_mask] = biofilm_color
    overlay[bacteria_mask] = bacteria_color
    
    # GT FOR VIZUALIZATION
    coloredMask = Image.open(coloredMaskPath).convert("RGBA")
    coloredMask = cropLineBelow(coloredMask, countPx=128)
    coloredMaskNP = np.array(coloredMask)

    black_mask = (coloredMaskNP[..., 0] == 0) & (coloredMaskNP[..., 1] == 0) & (coloredMaskNP[..., 2] == 0)
    new_data = coloredMaskNP.copy()
    new_data[..., 3][black_mask] = 0
    new_data[..., 3][~black_mask] = 255
    coloredMask = Image.fromarray(new_data, 'RGBA')
    coloredMaskNP = np.array(coloredMask)
    
    difference = np.zeros((height, width, 4), dtype=np.uint8)
    
    biofilm_diff_mask = (overlay == biofilm_color).all(axis=-1) & (coloredMaskNP == bacteria_color).all(axis=-1) 
    biofilm_diff_mask += (overlay != biofilm_color).all(axis=-1) & (coloredMaskNP == biofilm_color).all(axis=-1) 
    biofilm_diff_mask += (overlay == biofilm_color).all(axis=-1) & ~(
        ((coloredMaskNP == biofilm_color) 
         | (coloredMaskNP == bacteria_color)).all(axis=-1)
    )
    
    difference[biofilm_diff_mask] = difference_color
    
    bacteria_diff_mask = (overlay == bacteria_color).all(axis=-1) & (coloredMaskNP == biofilm_color).all(axis=-1) 
    bacteria_diff_mask += (overlay != bacteria_color).all(axis=-1) & (coloredMaskNP == bacteria_color).all(axis=-1) 
    bacteria_diff_mask += (overlay == bacteria_color).all(axis=-1) & ((coloredMaskNP != bacteria_color) & (coloredMaskNP != biofilm_color)).all(axis=-1)
    
    difference[bacteria_diff_mask] = difference_color
    
    origImg = Image.open(imgPath).convert("L")
    origImg = cropLineBelow(origImg, countPx=128)
    origImgNP = np.array(origImg) 
    
    # VIZUALIZATION ITSELF
    fig, axs = plt.subplots(1, 3, figsize=(15, 8))
    axs[0].imshow(origImgNP, cmap="gray")
    axs[0].imshow(coloredMask, alpha=0.7)   
    axs[0].axis('off')
    axs[1].imshow(origImgNP, cmap="gray")
    axs[1].imshow(overlay, alpha=0.7)   
    axs[1].axis('off')
    axs[2].imshow(origImgNP, cmap="gray")
    axs[2].imshow(difference, alpha=0.7)   
    axs[2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(outputDir, "difference_masks.png"), bbox_inches='tight')
    plt.show()
    
    # SAVE MASKS AND OVERLAYS
    origRGBA = Image.fromarray(origImgNP).convert("RGBA")  
    
    coloredMaskNP[..., 3] = 178  #  70%
    coloredMaskRGBA = Image.fromarray(coloredMaskNP, mode="RGBA")    
    composite = Image.alpha_composite(origRGBA, coloredMaskRGBA)
    composite.save(f"{outputDir}/{imgName}-single_and_biofilm_GTmask_RGB.png")
    
    overlay[..., 3] = 178  # 70%
    overlayRGBA = Image.fromarray(overlay, mode="RGBA")    
    composite = Image.alpha_composite(origRGBA, overlayRGBA)
    composite.save(f"{outputDir}/{imgName}-single_and_biofilm_PREDmask_RGBA.png")

    #plt.figure(figsize=(12, 12))
    #plt.imshow(origImgNP, cmap="gray")
    #plt.imshow(overlay, alpha=0.7)  
    #plt.axis('off')
    #plt.tight_layout()
    #plt.savefig(os.path.join(outputDir, "single_and_biofilm_PREDmaskOVERLAY.png"), bbox_inches='tight')
    
    #plt.figure(figsize=(12, 12))
    #plt.imshow(origImgNP, cmap="gray")
    #plt.imshow(coloredMask, alpha=0.7)   
    #plt.axis('off')
    #plt.tight_layout()
    #plt.savefig(os.path.join(outputDir, "single_and_biofilm_GTmaskOVERLAY.png"), bbox_inches='tight')
   

    # CONFUSION MATRIX
    confusionMatrix = np.zeros((3,3), dtype=np.uint64)
    confusionMatrix[0][0] = np.logical_and(biofilmGT == 1, biofilmPredictions == 1).sum()
    confusionMatrix[0][1] = np.logical_and(biofilmGT == 1, singlePredictions == 1).sum()
    confusionMatrix[0][2] = np.logical_and(biofilmGT == 1, backgroundPreds == 1).sum()

    confusionMatrix[1][0] = np.logical_and(singleGT == 1, biofilmPredictions == 1).sum()
    confusionMatrix[1][1] = np.logical_and(singleGT == 1, singlePredictions == 1).sum()
    confusionMatrix[1][2] = np.logical_and(singleGT == 1, backgroundPreds == 1).sum()

    confusionMatrix[2][0] = np.logical_and(backgroundGT == 1, biofilmPredictions == 1).sum()
    confusionMatrix[2][1] = np.logical_and(backgroundGT == 1, singlePredictions == 1).sum()
    confusionMatrix[2][2] = np.logical_and(backgroundGT == 1, backgroundPreds == 1).sum()
    
    print(f'---> {imgName} <---')
    print(f'CONFUSION MATRIX:\n', confusionMatrix)
    print(f'TOTAL pixels: {np.sum(confusionMatrix)}')
        
    # MULTICLASS METRICS
    y_true = np.stack([singleGT, biofilmGT, backgroundGT], axis=-1)
    y_pred = np.stack([singlePredictions, biofilmPredictions, backgroundPreds], axis=-1)
    y_true_flat = np.argmax(y_true, axis=-1).flatten()  # [0, 1, 2] — single, biofilm, background
    y_pred_flat = np.argmax(y_pred, axis=-1).flatten()
     
    print(f'\nP4 = {p4multicalss(confusionMatrix):.9f}')
    print(f'Accuracy = {accuracy_score(y_true_flat, y_pred_flat):.9f}')
    #print(f'Precision = {precision_score(y_true_flat, y_pred_flat, average=None)}') 
    #print(f'Recall = {recall_score(y_true_flat, y_pred_flat, average=None)}')
    print(f'IoU = {jaccard_score(y_true_flat, y_pred_flat, average=None)}\n')
    print(f'F1 = {f1_score(y_true_flat, y_pred_flat, average=None)}\n')

    with open(os.path.join(outputDir,"results.txt"), "a") as file:
        file.write(f'\n---> {imgName} <---\n')
        file.write(f'CONFUSION MATRIX: {confusionMatrix}\n')
        file.write(f'P4 = {p4multicalss(confusionMatrix):.9f}\n')
        file.write(f'Accuracy = {accuracy_score(y_true_flat, y_pred_flat)}\n')
        file.write(f'IoU = {jaccard_score(y_true_flat, y_pred_flat, average=None)}\n')
        file.write(f'F1 = {f1_score(y_true_flat, y_pred_flat, average=None)}\n')
        
    metrics_text = (
        f"P4 = {p4multicalss(confusionMatrix):.4f}\n"
        f"Accuracy = {accuracy_score(y_true_flat, y_pred_flat):.4f}\n"
        f"IoU = {jaccard_score(y_true_flat, y_pred_flat, average=None)}\n"
        f"F1 = {f1_score(y_true_flat, y_pred_flat, average=None)}"
    )
        
    return biofilmPredictions, singlePredictions, backgroundPreds, biofilmGT, singleGT, backgroundGT, confusionMatrix, metrics_text, imgName
