import os
import numpy as np
import cv2
from PIL import Image
from fastprogress import progress_bar
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
from . import ifnone, assert_in_list

__all__ = [
    'write_detection_features'
]

class DetectionCropDataset(Dataset):
    """
    Dataset that returns boxes crop from frames.
    """
    def __init__(self, mot_detections, img_path, img_size=128, thresh=0.0, 
                 preprocess=None, box_format='xywh', ext='PNG'):
        self.detects, self.img_path, self.ext = mot_detections, img_path, ext
        self.detects = self.detects[self.detects[6]>=thresh].reset_index(drop=True)
        
        self.preprocess = ifnone(preprocess, self.preprocess)(img_size)

        assert_in_list(box_format, ['xywh', 'xyxy'], 'box_format')
        # convert to xywh to save later. (consistent with MOT)
        if box_format == 'xyxy':
            self.detects[4] = self.detects[4] - self.detects[2]
            self.detects[5] = self.detects[5] - self.detects[3]

    def __getitem__(self, idx):
        x1, y1, w, h = self.detects.loc[idx, [2,3,4,5]].values.astype(int)
        img = self.imread(f'{str(self.detects.loc[idx, 0]).zfill(6)}.{self.ext}')
        img = img.crop([x1, y1, x1+w, y1+h])
        img = self.preprocess(img)
        return idx, img

    def __len__(self): return len(self.detects)

    def imread(self, p):
        return Image.open(os.path.join(self.img_path, p))

    def preprocess(self, img_size):
        return Compose([
            Resize(img_size),
            CenterCrop(img_size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

@torch.no_grad()
def write_detection_features(encoder, detections, destination='./features/', **dataset_kwargs):
    """
    Generate box features from detections and save at destination.
    
    a. Arguments:
        - encoder: feature encoder model
        - detections: detections csv file
        - destination: path to write the features
    """
    detections = detections.copy(deep=True)
    batch_size = int(dataset_kwargs.pop('batch_size'))
    num_workers = min(10, batch_size//4)
    
    detections['feat'] = ''
    crop_data = DetectionCropDataset(mot_detections=detections, thresh=0.0, **dataset_kwargs)
    crop_data = DataLoader(crop_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
                           pin_memory=True, drop_last=False)
    
    # reassign the xywh format converted detections file
    detections = crop_data.dataset.detects.copy(deep=True)

    os.makedirs(destination, exist_ok=True)

    encoder.cuda()
    encoder.eval()
    for i, (idxb, xb) in enumerate(progress_bar(crop_data)):
        feat = encoder(xb.cuda()).detach().cpu()
        for j,f in enumerate(feat):
            feat_path = os.path.join(destination, f'{str(idxb[j].item())}.npy')
            np.save(feat_path, f)
            detections.loc[idxb[j].item(), 'feat'] = os.path.abspath(feat_path)
    
    detections.columns = ['frame_id', 'track_id', 'x1', 'y1', 'w', 'h', 'confidence', 'na1', 'na2', 'na3', 'feat']
    detections_path = os.path.join(destination, 'detections.csv')
    detections.to_csv(detections_path, index_label='index')
    
    encoder.cpu()
    torch.cuda.empty_cache()
    del crop_data
    import gc
    gc.collect()


    print(f'Features is saved at {destination}')
    print(f'Detections file is saved at {detections_path}')