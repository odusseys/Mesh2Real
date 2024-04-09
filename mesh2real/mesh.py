import trimesh
import kaolin as kal
import xatlas 
import torch
import numpy as np


def rotate(tm, rotation_angle, rotation_direction):
  rot = trimesh.transformations.rotation_matrix(rotation_angle, rotation_direction)
  return tm.apply_transform(rot)
  
def load_mesh(path, rotations=[]):
  x = trimesh.load(path, force='mesh')
#   rotations = find_best_rotation(x, orientation_prompt) if orientation_prompt is not None else rotations if rotations is not None else []
  for [rotation_angle, rotation_direction] in rotations:
    x = rotate(x, rotation_angle, rotation_direction)

  vmapping, indices, uvs = xatlas.parametrize(x.vertices, x.faces)
  vertices = torch.tensor(x.vertices, device="cuda", dtype=torch.float32)
  faces = torch.tensor(x.faces.astype(np.int64), device="cuda")
  mesh = kal.rep.SurfaceMesh(vertices=vertices, faces=faces,
                           uvs=torch.tensor(uvs, device="cuda"),
                           face_uvs_idx=torch.tensor(indices.astype(np.int64), device="cuda")).to_batched()
  mesh.vertices = 0.5 * kal.ops.pointcloud.center_points(mesh.vertices, True)
  return mesh