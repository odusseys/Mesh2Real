import cv2
import numpy as np
from PIL import Image
import gc
import torch
import torchvision

def compute_edges(image):
  image = np.array(image)
  image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  image = cv2.Canny(image, 100, 200)
  return Image.fromarray(image)

def clear_cuda():
  with torch.no_grad():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def resize_image_array(img, size):
  transposed = torch.transpose(img, 0, 2)
  resized_transposed = torchvision.transforms.Resize(size, antialias=True)(transposed)
  return torch.transpose(resized_transposed, 0, 2)
