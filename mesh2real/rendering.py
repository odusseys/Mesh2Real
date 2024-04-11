import nvdiffrast
import torch
import numpy as np
from PIL import Image
import kaolin as kal
from .constants import IMAGE_SIZE, RENDERED_IMAGE_SIZE, TEXTURE_SIZE
import math
import torchvision
from .utils import compute_edges, resize_image_array

print("Initializing rendering context, this may take a while")
glctx = nvdiffrast.torch.RasterizeCudaContext(device=torch.device('cuda'))
print("Rendering context ready")

class ShadingOutputs:
  def __init__(self, shaded, roughness, metallic) -> None:
    self.shaded = shaded
    self.metallic = metallic
    self.roughness = roughness

class RenderOutput:
  def __init__(self, rendered, edges, depth, background_mask, object_mask, shading):
    self.rendered = rendered
    self.edges = edges
    self.depth = depth
    self.background_mask = background_mask
    self.object_mask = object_mask
    self.shading = shading

class Lighting:
  def __init__(self, direction=None, ambient_intensity=0.03, intensity=1.0):
    if direction is None:
      raise Exception("Direction must be provided")
    self.direction = direction
    self.intensity = intensity
    self.ambient_intensity = ambient_intensity

class ShadingMaps:
  def __init__(self, texture, roughness, metallic, ao=torch.tensor([0.03, 0.03, 0.03], device="cuda", requires_grad=False), requires_grad=False):
    self.texture = texture
    self.roughness = roughness
    self.metallic = metallic
    self.ao = ao
    self.texture.requires_grad_(requires_grad)
    self.roughness.requires_grad_(requires_grad)
    self.metallic.requires_grad_(requires_grad)
    self.metallic.requires_grad_(requires_grad)

  def to(self, device):
    self.texture = self.texture.to(device)
    self.roughness = self.roughness.to(device)
    self.metallic = self.metallic.to(device)

  def freeze_materials(self):
    self.roughness.requires_grad_(False)
    self.metallic.requires_grad_(False)
    self.texture.requires_grad_(True)

  def freeze_texture(self):
    self.metallic.requires_grad_(True)
    self.roughness.requires_grad_(True)
    self.texture.requires_grad_(False)

  def detach(self):
    return ShadingMaps(self.texture.detach(),
                       self.roughness.detach(),
                       self.metallic.detach(),
                       ao=self.ao.detach())
  

def init_maps(random_texture=False):
  if random_texture:
    texture = torch.rand((1, TEXTURE_SIZE, TEXTURE_SIZE, 3), device='cuda', requires_grad=True)
  else:
    texture = torch.full((1, TEXTURE_SIZE, TEXTURE_SIZE, 3), 1.0, device='cuda', requires_grad=True)
  roughness = torch.full((1, TEXTURE_SIZE, TEXTURE_SIZE, 1), 1.0, device='cuda', requires_grad=True)
  metallic = torch.full((1, TEXTURE_SIZE, TEXTURE_SIZE, 1), 0.0, device='cuda', requires_grad=True)
  return ShadingMaps(texture, roughness, metallic)



def make_camera(eye):
  return kal.render.camera.Camera.from_args(eye=torch.tensor(eye),
                                         at=torch.tensor([0., 0., 0.]),
                                         up=torch.tensor([0., 1., 0]),
                                         fov=math.pi * 30 / 180,
                                            near=0.1, far=10000.,
                                         width=IMAGE_SIZE,
                                            height=IMAGE_SIZE,
                                            device='cuda')

def polar_to_cartesian(r, phi, theta):
  y = r * math.cos(theta)
  z = r * math.sin(theta) * math.cos(phi)
  x = r * math.sin(theta) * math.sin(phi)
  return [x,y,z]

def polar_camera_and_light(r, phi, theta):
  eye = polar_to_cartesian(r, phi, theta)
  camera = make_camera(eye)
  eye = np.array(eye)
  eye_norm = np.sqrt(np.sum(eye * eye))
  light_direction = torch.tensor(eye / eye_norm, dtype=torch.float32).view(1, 1, 3).cuda()
  return camera, Lighting(direction=light_direction)

#
# All formulas taken from https://learnopengl.com/PBR/Lighting
#

def dot(X, Y):
  return torch.clamp(torch.sum(X * Y, dim=-1, keepdim=True), 0.0, 1.0)

def ggx_distribution(N, H, roughness):
  # GGX/Trowbridge-Reitz microfacet distribution
  alpha2 = roughness ** 4
  NH = dot(N, H)
  denom  = (NH * NH * (alpha2 - 1.0) + 1.0)
  denom = math.pi * denom * denom
  return alpha2/denom

def schlick_ggx_geometry(NV, roughness):
  r = roughness + 1
  k = r * r / 8
  return NV / (k + (1 - k) * NV)

def smith_geometry(N, V, L, roughness):
  NV = dot(N, V)
  NL = dot(N, L)
  return schlick_ggx_geometry(NV, roughness) * schlick_ggx_geometry(NL, roughness)

def shade_scene_pbr(mesh, maps, lighting, camera_eye, rast_out, rast_out_db):
  faces = mesh.faces.int().contiguous()
  face_uvs_idx = torch.squeeze(mesh.face_uvs_idx.int())
  uvs = mesh.uvs
  texc, texd = nvdiffrast.torch.interpolate(uvs, rast_out, face_uvs_idx, rast_db=rast_out_db, diff_attrs='all')
  def interpolate(x):
    return nvdiffrast.torch.interpolate(x, rast_out, faces, rast_db=rast_out_db, diff_attrs='all')[0]
  def texture(x):
    return nvdiffrast.torch.texture(x, texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=9)
  L = lighting.direction
  N = interpolate(mesh.vertex_normals)
  vertices_interpolated = interpolate(mesh.vertices)
  V = - vertices_interpolated + camera_eye
  V = V / torch.norm(V, dim=-1, keepdim=True)
  H = V + L
  H = H / torch.norm(H, dim=-1, keepdim=True)
  roughness = texture(maps.roughness)
  metallic = texture(maps.metallic)
  albedo = texture(maps.texture)

  # Fresnel-Schlick approximation
  F0 = metallic * albedo + (1 - metallic) * torch.tensor([0.04, 0.04, 0.04], device="cuda")
  VH = dot(V, H)
  F = F0 + (1 - F0) * ((1 - VH) ** 5)
  
  D = ggx_distribution(N, H, roughness)
  G = smith_geometry(N, V, L, roughness)

  NV = dot(N, V)
  NL = dot(N, L)
  specular = D * G * F / (4 * NL * NV + 0.0001)

  kd = (1 - F) * (1 - metallic)

  res = (kd * albedo / math.pi + specular) * lighting.intensity * NL

  res = res / (1 + res)
  res = res ** 0.4545
  shaded = torch.squeeze(res)
  return ShadingOutputs(shaded, roughness, metallic)

def render_scene(mesh, maps, cam, lighting, depth_slack=0.2,
                 resolution=RENDERED_IMAGE_SIZE, debug=False, background_color=0.0):
  vertices_camera = cam.extrinsics.transform(mesh.vertices)
  vertices_clip = cam.intrinsics.project(vertices_camera)
  faces = mesh.faces.int().contiguous()
  rast_out, rast_out_db = nvdiffrast.torch.rasterize(
      glctx, vertices_clip, faces, (resolution, resolution)
  )
  camera_eye = cam.extrinsics.cam_pos().reshape((1,1,3))

  depth = torch.sqrt(torch.sum((mesh.vertices - camera_eye) ** 2, 2))
  depth = depth.reshape(list(depth.shape) + [1])
  depth = nvdiffrast.torch.interpolate(depth, rast_out, faces, rast_db=rast_out_db, diff_attrs='all')[0]
  background = (depth == 0.0).squeeze(0)
  object_mask = depth > 0
  depth_max = torch.max(depth[object_mask])
  depth_min = torch.min(depth[object_mask])
  depth = torch.where(object_mask, 1.0 - (depth - depth_min) / (depth_max - depth_min), 0.0)
  # we allow a little bit of slack to avoid the back of the object being 0.0
  depth = torch.where(object_mask, (depth + depth_slack) / (1 + depth_slack), 0.0)
  depth = torch.squeeze(torch.cat((depth,depth,depth), 3))
  
  shading = shade_scene_pbr(mesh, maps, lighting, camera_eye, rast_out, rast_out_db)
  rendered = torch.where(background, background_color, shading.shaded)
  edge_input = resize_image_array(rendered, IMAGE_SIZE)

  shade_img = Image.fromarray((torch.squeeze(edge_input).detach().cpu().numpy() * 255.0).astype("uint8")).resize((IMAGE_SIZE, IMAGE_SIZE))
  edges = torch.tensor(np.array(compute_edges(shade_img)) * 255.0, device="cuda")


  return RenderOutput(
    rendered=rendered,
    edges=edges,
    depth=depth,
    background_mask=background, 
    object_mask=object_mask,
    shading=shading
  )


def pil_grid(images, max_horiz=np.iinfo(int).max):
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)
    for i, im in enumerate(images):
        h, v = i % n_horiz, i // n_horiz
        h_sizes[h] = max(h_sizes[h], im.size[0])
        v_sizes[v] = max(v_sizes[v], im.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
    for i, im in enumerate(images):
        im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
    return im_grid

def render_mesh_grid(mesh, maps, size=200):
  rend = []
  n = 9
  for i in range(n):
    cam, light_dir = polar_camera_and_light(1.5, 2 * math.pi * i/(n + 1), math.pi / 2 )
    output= render_scene(mesh, maps, cam, light_dir)
    image = Image.fromarray((output.rendered.detach().cpu().numpy() * 255.0).astype("uint8")).resize((size, size))
    rend.append(image)
  return pil_grid(rend, 3)

def make_mesh_gif(mesh, maps, path, size=256, n_frames=48):
  frames = []
  for i in range(n_frames):
    cam, light_dir = polar_camera_and_light(1.5, 2 * math.pi * i/(n_frames + 1), math.pi / 2)
    output = render_scene(mesh, maps, cam, light_dir)
    image = Image.fromarray((output.rendered.detach().cpu().numpy() * 255.0).astype("uint8")).resize((size, size))
    frames.append(image)
  frames[0].save(fp=path, format='GIF', append_images=frames[1:],
            save_all=True, duration=40, loop=0)
