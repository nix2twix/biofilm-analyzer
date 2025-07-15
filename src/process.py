import json
import sys
from PIL import Image
import torch
import os
from numpy import save
import numpy as np
from torch.utils.data import DataLoader
from dataset import TestDataset, splitDatasetInDirs
from preprocessing import binarizeMaskDir, cropLineBelow, slidingWindowPatch
from test import load_checkpoint, build_model, filter_masks_by_area_and_shape
import re
from skimage import measure
from cellpose import io, models
from cellpose.io import imread
from cellpose import plot
import matplotlib.pyplot as plt
import shutil 
from pathlib import Path
from skimage import measure
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    jaccard_score
)
from skimage.measure import label, regionprops
import logging
pattern = r'\.(\d+)_(\d+)\.png$'

OUTPUT_DIR = str(Path(__file__).parent / "tmp")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def filter_masks_by_area_and_shape(masks, probs, min_area, max_area, min_circularity=0.85):
    props = measure.regionprops(masks)
    areas = [prop.area for prop in props]
    filtered_probs = np.copy(probs)
    filtered_masks = np.copy(masks)
    for prop in props:
        ecc = prop.eccentricity 
        
        if (prop.area > max_area) or (prop.eccentricity < min_circularity) or (prop.area < min_area):
            filtered_masks[filtered_masks == prop.label] = 0
            filtered_probs[filtered_masks == prop.label] = 0.0

    return filtered_masks, filtered_probs

def processOneSEMimage(imagePath = None, imgMode = 'L', output_path = OUTPUT_DIR,
                           checkpointPath = None, 
                           min_area = 50,
                           max_area = 1155,
                           min_eccentricity = 0.85):
    
    img = Image.open(imagePath)
    img = img.convert(imgMode)
    imgName = os.path.basename(imagePath)
    logging.info(f"START PROCESSING {imgName}...")
    shutil.rmtree(output_path, ignore_errors=True) 
    os.makedirs(output_path)

    slidingWindowPatch(img, imgName, patch_size = (512, 512), save_dir = output_path,
                       visualize = False)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu" 
    test_dataset = TestDataset(
        image_dir=output_path,
        mode="test"
    )
    test_loader = DataLoader(test_dataset,
                             batch_size=2,
                             shuffle=False)

    model = build_model().to(device)   
    load_checkpoint(model, checkpointPath)
    model.eval()
    
    img = cropLineBelow(img, countPx=128)
    width, height = img.size
    probsCount = np.zeros((height, width), dtype=float)
    biofilmProbs = np.zeros((height, width), dtype=float)
    biofilmPredictions = np.zeros((height, width), dtype=float)
    
    probsCount = np.load("src/probsCount.npy")
    logging.info(f"START UNET++ PROCESSING...")
    with torch.no_grad():
        for i, (images, imgpaths) in enumerate(test_loader):
            images = images.to(device)
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
                logging.info(f'---> {img_name} <---')

    
    threshold = 0.5
    biofilmProbs = biofilmProbs / probsCount 
    biofilmPredictions = (biofilmProbs > threshold).astype(np.uint8) 
    
    origImg = Image.open(imagePath).convert("RGB")
    origImg = cropLineBelow(origImg, countPx=128)
    origImgNP = np.array(origImg) 
    
    cleaned_image = origImgNP.copy()
    cleaned_image[biofilmPredictions == 1] = 0 #black 
    logging.info(f"START CELLPOSE-SAM PROCESSING...")
    model_cp = models.CellposeModel(gpu=False)
    singlePredictions, flows, styles = model_cp.eval(cleaned_image, channels=[0, 0])

    singleProbs = 1 / (1 + np.exp(-singlePredictions))

    singlePredictions, singleProbs = filter_masks_by_area_and_shape(singlePredictions, 
                                                                    singleProbs, 
                                                                    min_area, 
                                                                    max_area, 
                                                                    min_eccentricity)
    
    # PREDS MATRIX
    singlePredictions = np.array(singlePredictions != 0, dtype=np.uint8)
    
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
    
    origImg = Image.open(imagePath).convert("L")
    origImg = cropLineBelow(origImg, countPx=128)
    origImgNP = np.array(origImg) 
    
    # SAVE 
    origRGBA = Image.fromarray(origImgNP).convert("RGBA")  
    alpha_mask = (overlay[..., :3] != 0).any(axis=-1)
    overlay[alpha_mask, 3] = 178  # 70%
    overlayRGBA = Image.fromarray(overlay, mode="RGBA")    
    composite = Image.alpha_composite(origRGBA, overlayRGBA)
    composite.save(f"{output_path}/{imgName}-result.png")
    
    labeled_bacteria = measure.label(singlePredictions)
    bacteria_count = labeled_bacteria.max()
    biofilm_area = np.sum(biofilm_mask)
    result_info = {
    "biofilm_area": int(np.sum(biofilm_mask)),
    "bacteria_count": int(bacteria_count)
    }
    with open("result_stats.json", "w") as f:
        json.dump(result_info, f)
    logging.info(f"PROCESSED SUCCESSFULLY!")
    return composite

if __name__ == "__main__":
    if len(sys.argv) > 1:
        params = json.loads(sys.argv[1])
        min_area = params.get("min_area")
        max_area = params.get("max_area")
        min_eccentricity = params.get("min_eccentricity")

        resultImage = processOneSEMimage(imagePath="input_image.bmp",
                               output_path = "processingResults",
                               checkpointPath="src/final_model_epoch_350.pth",
                               min_area = min_area,
                               max_area = max_area,
                               min_eccentricity = min_eccentricity              
        )
        if os.path.exists("output_image.bmp"):
            os.remove("output_image.bmp")
        resultImage.save("output_image.bmp")