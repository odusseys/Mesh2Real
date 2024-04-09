import torch
from PIL import Image
import numpy as np
from featup.util import norm
from sklearn.decomposition import PCA
import torchvision.transforms as T
import cv2
import math
from scipy.interpolate import griddata
from skimage.color import rgb2gray
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1

def make_upsampler():
    return torch.hub.load("mhamilton723/FeatUp", 'dinov2').cuda().eval()


def warp_image(image0, image1, target, order=3):
  image0 = np.array(image0)
  image1 = np.array(image1)

  # --- Convert the images to gray level: color is not supported.
  image0gray = rgb2gray(image0)
  image1gray = rgb2gray(image1)


  # --- Compute the optical flow
  v, u = optical_flow_tvl1(image0gray, image1gray, prefilter=True)

  # --- Use the estimated optical flow for registration

  nr, nc = image0gray.shape

  row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                      indexing='ij')

  def warp_channel(channel):
      channel = channel / 255.0
      return warp(channel, np.array([row_coords + v, col_coords + u]),
                      mode='edge', order=order)
  target = np.array(target)
  r = warp_channel(target[:,:,0])
  g = warp_channel(target[:,:,1])
  b = warp_channel(target[:,:,2])
  target_warp = Image.fromarray((np.stack([r,g,b], axis=2) * 255).astype(np.uint8))
  return target_warp

FEATURE_MAP_SIZE = 256
N_FEATURES = 384

transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    norm
])

def get_featup_features(upsampler, images):
  with torch.no_grad():
    images = [transform(image) for image in images]
    images = torch.stack(images).to(dtype=torch.float32, device="cuda")
    res = upsampler(images)
    del images
    return res

def reduce_dimension(f1, f2):
  f1 = f1.permute((1, 2, 0)).cpu().numpy().reshape((FEATURE_MAP_SIZE * FEATURE_MAP_SIZE, N_FEATURES))
  f2 = f2.permute((1, 2, 0)).cpu().numpy().reshape((FEATURE_MAP_SIZE * FEATURE_MAP_SIZE, N_FEATURES))
  features = np.concatenate([f1, f2], axis=0)
  reduced = PCA(n_components=3).fit_transform(features)
  return reduced[:f1.shape[0]], reduced[f1.shape[0]:]

def norm_features(f):
  f = f.reshape((FEATURE_MAP_SIZE, FEATURE_MAP_SIZE, 3))
  f = f - np.min(f)
  f = f / np.max(f)
  f = (f * 255).astype(np.uint8)
  # normalize histograms for each channel
  for i in range(3):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    f[:,:,i] = clahe.apply(f[:,:,i])
  return f

def semantic_registration(image0, image1):
    [f1, f2] = get_featup_features(upsampler, [image0, image1])
    f1, f2 = reduce_dimension(features[0], features[1], 3)
    f1 = norm_features(f1)
    f2 = norm_features(f2)
    return warp_image(Image.fromarray(f1.astype(np.uint8)), Image.fromarray(f2.astype(np.uint8)), image1)

def edge_registration(pipeline, image, edges):
    def denoise(init_image, strength=1,
            controlnet_conditioning_scale=0.5, ip_adapter_scale=0.5):
        return pipeline(
            prompt="",
            edges=edges,
            image=init_image,
            strength=strength,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            ip_adapter_image=image,
            ip_adapter_scale=ip_adapter_scale
        ).images[0]
        
    raw_correction = denoise(warped,
                        controlnet_conditioning_scale=0.9, strength=1, ip_adapter_scale=1)
    
    warped = semantic_registration(image, raw_correction)
    return denoise("", warped,
                        controlnet_conditioning_scale=0.3, strength=0.5, ip_adapter_scale=1)
