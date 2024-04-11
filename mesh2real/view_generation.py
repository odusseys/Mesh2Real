from diffusers import DiffusionPipeline, ControlNetModel
import torch
import rembg
import numpy as np
from PIL import Image
from .rendering import render_scene, polar_camera_and_light, init_maps
from .constants import IMAGE_SIZE
from .pipelines import Mesh2RealDiffuser
import math

ANGLES_DEG = [
    [30, 20], [90, -10], [150, 20], [210, -10], [270, 20], [330, -10]
]

ANGLES_RAD = [[x * math.pi / 180, y * math.pi / 180] for [x, y] in ANGLES_DEG]

DEPTH_IMAGE_SIZE = 320    

R = 1.5

def make_depth_grid(depths):
  depth_grid = Image.new('RGB', (DEPTH_IMAGE_SIZE * 2, DEPTH_IMAGE_SIZE * 3), color='white')
  for (n, d) in enumerate(depths):
    i = n % 2
    j = n //2
    depth_grid.paste(d, (i * DEPTH_IMAGE_SIZE, j * DEPTH_IMAGE_SIZE))
  return depth_grid

def make_view_inputs(mesh, maps, elevation, azimuth):
    theta_view = math.pi / 2 + elevation
    phi_view = azimuth
    cam, light = polar_camera_and_light(R, phi_view, theta_view)
    output = render_scene(mesh, maps, cam, light, depth_slack=0.0)
    depth = output.depth.cpu().detach().numpy()

    depth = 1.0 - depth # zero123++ works with inverted depth for some reason
    depth = Image.fromarray((depth * 255.0).astype(np.uint8)).resize((DEPTH_IMAGE_SIZE,DEPTH_IMAGE_SIZE))
    return depth, output.edges

def prepare_zero123_inputs(mesh):
  phi = 0
  theta = math.pi / 2
  ref_cam, ref_light = polar_camera_and_light(R, phi, theta)
  maps = init_maps()
  output = render_scene(mesh, maps, ref_cam, ref_light, depth_slack=0.0, resolution=IMAGE_SIZE)
  depth = output.depth.cpu().detach().numpy()
  condition_edges = Image.fromarray((output.edges.cpu().detach().numpy() * 255.0).astype(np.uint8))
  condition_depth = Image.fromarray((depth * 255.0).astype(np.uint8))
  view_depths = []
  view_edges = []
  for [azimuth, elevation] in ANGLES_RAD:
    depth, edges = make_view_inputs(mesh, maps, elevation, azimuth)
    view_depths.append(depth)
    view_edges.append(edges)

  depth_grid = make_depth_grid(view_depths)
  return condition_edges, condition_depth, depth_grid, view_edges


def breakdown_view_grid(grid):
  views = []
  w = grid.width
  h = grid.height
  for j in range(3):
    for i in range(2):
      cropped = grid.crop((i * w/2, j * h/3, (i + 1) * w / 2, (j + 1) * h/3))
      view = rembg.remove(cropped, bgcolor=[255, 255, 255, 1]).convert("RGB")
      views.append(view.resize((IMAGE_SIZE, IMAGE_SIZE)))
  return views

def generate_views(zero123_pipeline, image, image_depth, depth_grid, num_inference_steps=100):
  image_array = np.array(image)
  image_depth_array = np.array(image_depth)
  image_array[image_depth_array == 0.0] = 255 # FIXME: directly render the image with a white bg
  image = Image.fromarray(image_array)
  with torch.no_grad():
    image = image.resize((320, 320))
    depth = depth.resize((320 * 2, 320 * 3))
    result = zero123_pipeline(image, depth_image=depth_grid, num_inference_steps=num_inference_steps).images[0]
    views = breakdown_view_grid(result)
    return views

class ViewGenerator():

  def __init__(self):
    pipe = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16
    )
    pipe.add_controlnet(ControlNetModel.from_pretrained(
        "sudo-ai/controlnet-zp11-depth-v1", torch_dtype=torch.float16
    ), conditioning_scale=0.75)
    self.pipe = pipe

  
  def generate_views(self, diffusion_pipeline: Mesh2RealDiffuser, mesh, prompt, negative_prompt = None,  num_inference_steps=100):
    with torch.no_grad():
      condition_edges, condition_depth, depth_grid, view_edges = prepare_zero123_inputs(mesh)
      initial_image = diffusion_pipeline(prompt=prompt, edges=condition_edges, negative_prompt=negative_prompt, fix_edges=True)
      image_array = np.array(initial_image)
      image_depth_array = np.array(condition_depth)
      image_array[image_depth_array == 0.0] = 255 # FIXME: directly render the image with a white bg
      image = Image.fromarray(image_array).resize((320, 320))
      image = image.resize((DEPTH_IMAGE_SIZE, DEPTH_IMAGE_SIZE))
      depth = depth.resize((DEPTH_IMAGE_SIZE * 2, DEPTH_IMAGE_SIZE * 3))
      result = self.pipe(image, depth_image=depth_grid, num_inference_steps=num_inference_steps).images[0]
      views = breakdown_view_grid(result)
      
      fixed = []
      for view, edges in zip(views, view_edges):
         f = diffusion_pipeline.fix_edges(view, edges)
         fixed.append(f)
      return initial_image, fixed
    
  def to(self, device):
    self.pipe.to(device)
    