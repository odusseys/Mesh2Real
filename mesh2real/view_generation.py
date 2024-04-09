from diffusers import DiffusionPipeline, ControlNetModel
import torch
import rembg
import numpy as np
from PIL import Image
from .rendering import render_scene, polar_camera_and_light
from .constants import IMAGE_SIZE, RENDERED_IMAGE_SIZE

def make_zero123_pipeline():
    zero123_pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16
    )
    zero123_pipeline.add_controlnet(ControlNetModel.from_pretrained(
        "sudo-ai/controlnet-zp11-depth-v1", torch_dtype=torch.float16
    ), conditioning_scale=0.75)
    return zero123_pipeline.to("cuda")

def breakdown_view_grid(grid):
  views = []
  w = grid.width
  h = grid.height
  for j in range(3):
    for i in range(2):
      cropped = grid.crop((i * w/2, j * h/3, (i + 1) * w / 2, (j + 1) * h/3)).resize((IMAGE_SIZE, IMAGE_SIZE))
      view = rembg.remove(cropped, bgcolor=[255,255,255, 1]).convert("RGB")
      views.append(view)
  return views


def generate_views(zero123_pipeline, image, image_depth, depth, num_inference_steps=100):
  image_array = np.array(image)
  image_depth_array = np.array(image_depth)
  image_array[image_depth_array == 0.0] = 255
  image = Image.fromarray(image_array)
  with torch.no_grad():
    image = image.resize((320, 320))
    depth = depth.resize((320 * 2, 320 * 3))
    result = zero123_pipeline(image, depth_image=depth, num_inference_steps=num_inference_steps).images[0]
    views = breakdown_view_grid(result)
    return result, views


ANGLES_DEG = [
    [30, 20], [90, -10], [150, 20], [210, -10], [270, 20], [330, -10]
]

ANGLES_RAD = [[x * math.pi / 180, y * math.pi / 180] for [x, y] in ANGLES_DEG]

DEPTH_IMAGE_SIZE = 320

def prepare_zero123_inputs(mesh, maps, r, theta, phi):
  ref_cam, ref_light = polar_camera_and_light(r, phi, theta)
  output = render_scene(mesh, maps, ref_cam, ref_light, depth_slack=0.0)
  depth = output.depth.cpu().detach().numpy()
  condition = Image.fromarray((output.rendered.cpu().detach().numpy() * 255.0).astype(np.uint8))
  condition_edges = Image.fromarray((output.edges.cpu().detach().numpy() * 255.0).astype(np.uint8))
  condition_depth = Image.fromarray((depth * 255.0).astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE))
  depths = []
  for [azimuth, elevation] in ANGLES_RAD:
    theta_view = theta + elevation
    phi_view = phi + azimuth
    cam, light = polar_camera_and_light(r, phi_view, theta_view)
    output = render_scene(mesh, maps, cam, light, depth_slack=0.0)
    depth = output.depth.cpu().detach().numpy()

    depth = 1.0 - depth # zero123++ works with inverted depth for some reason
    depth = Image.fromarray((depth * 255.0).astype(np.uint8)).resize((DEPTH_IMAGE_SIZE,DEPTH_IMAGE_SIZE))
    depths.append(depth)

  depth_grid = Image.new('RGB', (DEPTH_IMAGE_SIZE * 2, DEPTH_IMAGE_SIZE * 3), color='white')
  for (n, d) in enumerate(depths):
    i = n % 2
    j = n //2
    depth_grid.paste(d, (i * DEPTH_IMAGE_SIZE, j * DEPTH_IMAGE_SIZE))
  return condition, condition_edges, condition_depth, depth_grid
