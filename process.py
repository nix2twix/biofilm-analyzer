import json
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
#from cellpose import io, models
#from cellpose.io import imread
#from cellpose import plot
import matplotlib.pyplot as plt
from gradio_client import Client, file, handle_file
import shutil 
pattern = r'\.(\d+)_(\d+)\.png$'
import tifffile
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

from pathlib import Path
OUTPUT_DIR = str(Path(__file__).parent / "tmp")

def processOneSEMimage(imagePath = None, imgMode = 'L', output_path = OUTPUT_DIR,
                      configPath = None, checkpointPath = None):
    
    img = Image.open(imagePath)
    img = img.convert(imgMode)
    imgName = os.path.basename(imagePath)

    shutil.rmtree(output_path, ignore_errors=True) 
    os.makedirs(output_path)

    #slidingWindowPatch(img, imgName, patch_size = (512, 512), save_dir = output_path,
                      # visualize = False)

    with open(configPath) as f:
        cfg = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu" 
    test_dataset = TestDataset(
        image_dir=output_path,
        mode="test"
    )
    test_loader = DataLoader(test_dataset,
                             batch_size=2,
                             shuffle=False)

    model = build_model(cfg["model"]).to(device)   
    load_checkpoint(model, checkpointPath)
    model.eval()
    
    img = cropLineBelow(img, countPx=128)
    width, height = img.size
    probsCount = np.zeros((height, width), dtype=float)
    biofilmProbs = np.zeros((height, width), dtype=float)
    biofilmPredictions = np.zeros((height, width), dtype=float)
    
    probsCount = np.load("probsCount.npy")
    
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
                print(f'---> {img_name} <---')

    
    threshold = 0.5
    biofilmProbs = biofilmProbs / probsCount 
    biofilmPredictions = (biofilmProbs > threshold).astype(np.uint8) 
    
    origImg = Image.open(imagePath).convert("RGB")
    origImg = cropLineBelow(origImg, countPx=128)
    origImgNP = np.array(origImg) 
    
    cleaned_image = origImgNP.copy()
    cleaned_image[biofilmPredictions == 1] = 0 #black 
    cleaned_image = Image.fromarray(cleaned_image)
    cleaned_image.save(os.path.join(output_path, 'blackBF.png'))

    client = Client("mouseland/cellpose")
    result = client.predict(
      filepath=[handle_file(os.path.join(output_path, 'blackBF.png'))],
      resize=1000,
      max_iter=250,
      flow_threshold=0.4,
      cellprob_threshold=0,
      api_name="/cellpose_segment"
    )

    #model_cp = models.CellposeModel(gpu=False)
    #singlePredictions, flows, styles = model_cp.eval(cleaned_image, channels=[0, 0])

    flows_path = result[1] 
    flows_image = Image.open(flows_path)
    flows_array = np.array(flows_image) 
    dP_x = (flows_array[:, :, 0].astype(np.float32) - 128) / 127
    dP_y = (flows_array[:, :, 1].astype(np.float32) - 128) / 127
    dP = np.stack([dP_x, dP_y], axis=-1)  

    cellprob = (flows_array[:, :, 2].astype(np.float32) / 255 * 12) - 6

    masks = tifffile.imread(result[2]['value'])
    singlePredictions = (masks > 0).astype(np.uint8)
    singleProbs = 1 / (1 + np.exp(-singlePredictions))

    #singlePredictions, singleProbs = filter_masks_by_area_and_shape(singlePredictions, singleProbs)
    
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

    return composite

if __name__ == "__main__":

    resultImage = processOneSEMimage(imagePath="input_image.bmp",
                           output_path = "./biofilm-analyzer/tmp/processingResults",
                           configPath = "./test_config.json",
                           checkpointPath = "./final_model_epoch_300.pth"
    )
    #resultImage.show()
    resultImage.save("output_image.bmp")