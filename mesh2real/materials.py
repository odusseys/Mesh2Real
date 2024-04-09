import torch
from PIL import Image
import torch.nn as nn
import torchvision.transforms as T
from featup.util import norm
import numpy as np

N_FEATURES = 384
IMAGE_SIZE = 256
N_MAPS=2
PARAM_NAMES = ["roughness", "metallic"]

IMAGE_TRANSFORM = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    norm
])

def normalize_image(image, size=IMAGE_SIZE):
  l = min(image.width, image.height)
  return image.crop((0,0,l,l)).resize((size,size))


def make_material_map_images(maps, size=256):
  maps_processed = []
  for i in range(2):
    print(PARAM_NAMES[i])
    map = torch.clamp(maps[:,:,i], 0, 1)
    img = (map.cpu().detach().numpy() * 255.0).astype(np.uint8)
    maps_processed.append(img)
  return Image.fromarray(maps_processed[0]).resize((size, size)), Image.fromarray(maps_processed[1]).resize((size, size))


class DepthwiseConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False):
    super().__init__()
    self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=bias)
    self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

  def forward(self, x):
    return self.pointwise(self.depthwise(x))

class ResidualBlock(nn.Module):
  def __init__(self, n_features):
    super().__init__()
    self.weight_units = nn.Sequential(
        DepthwiseConv2d(n_features, n_features, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(n_features),
        nn.LeakyReLU(),
        DepthwiseConv2d(n_features, n_features, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(n_features),
        nn.LeakyReLU(),
    )

  def forward(self, x):
    return x + self.weight_units(x)

class ProjectionLayer(nn.Module):
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(input_dim, output_dim, 1, bias=False),
      nn.BatchNorm2d(output_dim),
      nn.LeakyReLU(),
    )

  def forward(self, x):
    return self.layers(x)


class MaterialPredictionHead(nn.Module):
  def __init__(self):
    super().__init__()
    self.image_transform = nn.Sequential(
        ProjectionLayer(3, 64),
        ResidualBlock(64),
        ResidualBlock(64),
    )
    self.feature_transform = nn.Sequential(
        ProjectionLayer(N_FEATURES, 128),
        ResidualBlock(128),
        ProjectionLayer(128, 64),
        ResidualBlock(64),
    )
    self.head = nn.Sequential(
        ProjectionLayer(128, 64),
        ResidualBlock(64),
        ProjectionLayer(64, 32),
        ResidualBlock(32),
        ProjectionLayer(32, 16),
        ResidualBlock(16),
        ProjectionLayer(16, N_MAPS),
    )

  def forward(self, pixel_values, feature_maps):
    img_transformed = self.image_transform(pixel_values)
    feats_transformed = self.feature_transform(feature_maps)
    features_fused = torch.cat([img_transformed, feats_transformed], dim=1)
    return self.head(features_fused).permute((0, 2, 3, 1))

class MaterialPredictor(nn.Module):
  def __init__(self, head, upsampler = None, device="cuda"):
    super().__init__()
    self.device = device
    self.upsampler = upsampled if upsampler is not None else torch.hub.load("mhamilton723/FeatUp", 'dinov2').to(device).eval()
    self.head = head
  
  def forward(self, images):
    images = [normalize_image(image) for image in images]
    with torch.no_grad():
      data = [IMAGE_TRANSFORM(image) for image in images]
      data = torch.stack(data).to(dtype=torch.float32, device=self.device)
      features = self.upsampler(data)
      del data
      pixel_values = torch.stack([T.functional.pil_to_tensor(x) for x in images]).to(
          self.device, torch.float32, memory_format=torch.channels_last) / 255.0    
    res = self.head(pixel_values, features)
    return torch.clip(res, 0, 1)
  
  def from_checkpoint(path, upsampler=None, device="cuda"):
    head = MaterialPredictionHead().to(device)
    head.load_state_dict(torch.load(path))
    return MaterialPredictor(head, upsampler, device)