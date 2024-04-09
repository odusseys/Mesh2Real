import torch
import numpy as np
from .view_generation import ANGLES_RAD
from .constants import IMAGE_SIZE, RENDERED_IMAGE_SIZE
from .rendering import init_maps

def blur(image, n):
  image = torch.transpose(image, 3, 1)
  kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1).cuda() / 16.0
  for i in range(n):
    image = torch.nn.functional.conv2d(image, kernel, padding=1, groups=3)
  image = torch.transpose(image, 3, 1)
  return image

def texture_loss(image):
  image = blur(image, 3)
  image = torch.transpose(image, 3, 1)
  sobel_x = torch.tensor([[1, 2, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1).cuda()
  sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1).cuda()
  sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1).cuda()
  grad_x = torch.nn.functional.conv2d(image, sobel_x, padding=1, groups=3)
  grad_y = torch.nn.functional.conv2d(image, sobel_y, padding=1, groups=3)
  l = torch.mean(grad_x ** 2 + grad_y ** 2)
  return l

def angle_generator(phi_range=[0, 0.1], theta_range=[0, 2], n=8):
  phi_values = np.linspace(phi_range[0], phi_range[1], n)
  theta_values = np.linspace(theta_range[0], theta_range[1], n)

  phi_index = 0
  theta_index = 0

  def next_angles():
    nonlocal phi_index
    nonlocal theta_index
    nonlocal phi_values
    nonlocal theta_values
    phi = phi_values[phi_index]
    theta = theta_values[theta_index]
    phi_index += 1
    if phi_index>= n:
      phi_index = 0
      theta_index +=1
    if theta_index >= n:
      theta_index = 0
    return phi, theta

  return next_angles


lambda_image = 0


def bake_initial_texture(mesh, main_view, views, initial_maps=None, num_epochs=50, lr=8.0):
  maps = initial_maps if initial_maps is not None else init_maps()
  maps.freeze_materials()
  texture = maps.texture
  plot_interval = num_epochs // 10 if num_epochs > 10 else 1
  optimizer = torch.optim.SGD(params=[texture], lr=lr)
  loss_with_decay = None
  loss_decay = 0.9
  i = 0

  for epoch in range(num_epochs):
    if i == 0:
      phi, theta = 0, math.pi/2
      condition = main_view
    else:
      [phi, elevation] = ANGLES_RAD[i - 1]
      theta = elevation + math.pi / 2
      condition = views[i - 1]
    i = (i + 1) % (len(ANGLES_RAD) + 1)
    cam, lighting = polar_camera_and_light(1.5, phi, theta)
    rendered = render_scene(mesh, maps, cam, lighting).rendered
    real = torch.tensor(np.array(condition.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255).to("cuda")
    # display_array_image(rendered)
    # display_array_image(real)
    loss = ((rendered - real) ** 2).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
      torch.clamp_(texture, 0, 1)
    loss_with_decay = loss if loss_with_decay is None else loss_with_decay * loss_decay + loss * (1 - loss_decay)
  return maps.detach()

def bake_materials(mesh, maps, num_epochs=28, lr=8.0):
  maps.freeze_texture()

  plot_interval = num_epochs // 10 if num_epochs > 10 else 1
  optimizer = torch.optim.SGD(params=[maps.ka, maps.kd, maps.ks, maps.alpha], lr=lr)
  loss_with_decay = None
  loss_decay = 0.9
  i = 0

  for epoch in range(num_epochs):
    if i == 0:
      phi, theta = 0, math.pi/2
    else:
      [phi, elevation] = ANGLES_RAD[i - 1]
      theta = elevation + math.pi / 2
    i = (i + 1) % (len(ANGLES_RAD) + 1)
    cam, lighting = polar_camera_and_light(1.5, phi, theta)
    output = render_scene(mesh, maps, cam, lighting)
    pil_image = Image.fromarray((output.rendered.detach().cpu().numpy() * 255.0).astype("uint8"))
    with torch.no_grad():
      pred = matformer([pil_image])[0]
      # display_array_image(pred[:,:,3])
      pred = resize_image_array(pred, RENDERED_IMAGE_SIZE)
    shading = output.shading

    materials = torch.squeeze(torch.stack([shading.ka, shading.kd, shading.ks, shading.alpha], axis=-1))
    loss = ((pred - materials) ** 2).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    with torch.no_grad():
      torch.clamp_(maps.ks, 0, 1)
      torch.clamp_(maps.kd, 0, 1)
      torch.clamp_(maps.ka, 0, 1)
      torch.clamp_(maps.alpha, 0, 1)

    loss_with_decay = loss if loss_with_decay is None else loss_with_decay * loss_decay + loss * (1 - loss_decay)
  maps.normalize()
  return maps.detach()

def refine_texture(prompt, mesh, maps, num_epochs=100, lr=1.0, strength=0.8,
                  r_range=[0.7, 0.8], phi_range=[0, 0.1], theta_range=[0, 2], ):
  maps.freeze_materials()
  plot_interval = num_epochs // 10 if num_epochs > 10 else 1
  optimizer = torch.optim.SGD(params=[maps.texture], lr=lr)
  loss_with_decay = None
  loss_decay = 0.9

  next_angle = angle_generator(phi_range, theta_range)

  for epoch in range(num_epochs):
    print("epoch", epoch + 1)
    clear_cuda()
    phi, theta = next_angle()
    cam, lighting = polar_camera_and_light(r_range[0], phi, theta)
    output = render_scene(mesh, maps, cam, lighting)
    pil_image = Image.fromarray((output.rendered.detach().cpu().numpy() * 255.0).astype("uint8")).resize((IMAGE_SIZE, IMAGE_SIZE))
    pil_edges = Image.fromarray((output.edges.detach().cpu().numpy() * 255.0).astype("uint8")).resize((IMAGE_SIZE, IMAGE_SIZE))
    pil_depth= Image.fromarray((output.depth.detach().cpu().numpy() * 255.0).astype("uint8")).resize((IMAGE_SIZE, IMAGE_SIZE))
    real_img = denoise(prompt, pil_image, pil_edges, pil_depth, strength=strength)
    real = torch.tensor(np.array(real_img) / 255).to("cuda")
    loss = ((output.rendered - real) ** 2).sum()
    loss += lambda_image * texture_loss(maps.texture)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
      torch.clamp_(maps.texture, 0, 1)
    loss_with_decay = loss if loss_with_decay is None else loss_with_decay * loss_decay + loss * (1 - loss_decay)
    print("epoch", epoch, "loss:", loss_with_decay.detach().cpu().numpy())
    if (epoch + 1) % plot_interval == 0:
      display(pil_image.resize((300, 300)))
      display(real_img.resize((300, 300)))
      render_mesh_grid(mesh, maps)
  return maps.detach()
