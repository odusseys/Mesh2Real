from lib2to3 import fixer_base
import torch
from transformers import CLIPVisionModelWithProjection
from PIL import Image
import numpy as np
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image, EulerDiscreteScheduler, ControlNetModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from .registration import edge_registration
from .utils import clear_cuda
from diffusers.image_processor import IPAdapterMaskProcessor

def make_img2img_pipe():
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to("cuda")

    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_lora.safetensors" # Use the correct ckpt for your step setting!

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16
    )

    lora_weights = hf_hub_download(repo, ckpt)

    pipe = AutoPipelineForImage2Image.from_pretrained(
        base, controlnet=controlnet, image_encoder=image_encoder, torch_dtype=torch.float16, variant="fp16"
        )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.unet.to(memory_format=torch.channels_last)
    
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.bin")

    pipe.load_lora_weights(lora_weights)
    pipe.fuse_lora()
    res = pipe.to("cuda")
    del pipe
    clear_cuda()
    return res

def make_text2img_pipe(img2img_pipeline):
    # controlnet = ControlNetModel.from_pretrained(
    #     "diffusers/controlnet-canny-sdxl-1.0",
    #     variant="fp16",
    #     use_safetensors=True,
    #     torch_dtype=torch.float16,
    # )

    # base = "stabilityai/stable-diffusion-xl-base-1.0"
    # repo = "ByteDance/SDXL-Lightning"
    # ckpt = "sdxl_lightning_4step_lora.safetensors" # Use the correct ckpt for your step setting!

    # lora_weights = hf_hub_download(repo, ckpt)

    # pipe = AutoPipelineForText2Image.from_pretrained(
    #     base, controlnet=controlnet, torch_dtype=torch.float16, variant="fp16"
    # )
    # pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    # pipe.unet.to(memory_format=torch.channels_last)
    
    # pipe.load_lora_weights(lora_weights)
    # pipe.fuse_lora()
    # pipe.enable_model_cpu_offload()

    pipe = AutoPipelineForText2Image.from_pipe(img2img_pipeline)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.bin")

    return pipe

DEFAULT_NEGATIVE_PROMPT = "low quality, worst quality"
IMAGE_SIZE = 1024
BLANK_IMAGE = Image.fromarray(np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 0).astype(np.uint8))

class Mesh2RealDiffuser():
    def __init__(self, img2img_pipeline, text2img_pipeline):
        super().__init__()
        self.img2img_pipeline = img2img_pipeline
        self.text2img_pipeline = text2img_pipeline
    
    def build():
        img2img_pipeline = make_img2img_pipe().to("cuda")
        text2img_pipeline = make_text2img_pipe(img2img_pipeline).to("cuda")
        return Mesh2RealDiffuser(img2img_pipeline, text2img_pipeline)

    def to(self, device):
        self.img2img_pipeline.to(device)
        self.text2img_pipeline.to(device)

    def _text2img(self, prompt, edges, negative_prompt, controlnet_scale, ip_adapter_image, ip_adapter_scale, masks):
        clear_cuda()
        self.text2img_pipeline.set_ip_adapter_scale(ip_adapter_scale)
        return self.text2img_pipeline(
            prompt, 
            image=edges, 
            negative_prompt=negative_prompt, 
            controlnet_conditioning_scale=controlnet_scale,
            num_inference_steps=4,
            guidance_scale=0.0,
            ip_adapter_image=ip_adapter_image,
            cross_attention_kwargs=masks
        ).images[0]

    def _img2img(self, prompt, edges, negative_prompt, image, strength, controlnet_scale, ip_adapter_image, ip_adapter_scale, masks):
        if strength not in [0.0, 0.25, 0.5, 1.0]:
                raise ValueError("Strength must be one of [0.0, 0.25, 0.5, 1.0]")
            
        if image.width != IMAGE_SIZE or image.height != IMAGE_SIZE:
            raise ValueError("image size must be 1024x1024")
        clear_cuda()

        with torch.no_grad():
            self.img2img_pipeline.set_ip_adapter_scale(ip_adapter_scale)
            return self.img2img_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image, 
                control_image=edges, 
                strength=strength,
                ip_adapter_image=ip_adapter_image,
                controlnet_conditioning_scale=controlnet_scale,
                num_inference_steps=4,
                guidance_scale=0.0,
                cross_attention_kwargs=masks
            ).images[0]
        
    def __call__(self, 
        prompt, 
        edges, 
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        image=None, 
        ip_adapter_image=None, 
        ip_adapter_scale = None,
        controlnet_scale = 0.3,
        strength=0.5,
        fix_edges=False
    ):
        if edges is None:
            raise ValueError("Edges must be provided")
        if edges.width != IMAGE_SIZE or edges.height != IMAGE_SIZE:
            raise ValueError("Edges size must be 1024x1024")
        if ip_adapter_image:
            if ip_adapter_image.width != IMAGE_SIZE or ip_adapter_image.height != IMAGE_SIZE:
                raise ValueError("ip_adapter_image size must be 1024x1024")
        with torch.no_grad():
            masks = None
            res = None
            if ip_adapter_image is None:
                ip_adapter_image = BLANK_IMAGE
                ip_adapter_scale = 0.0
                processor = IPAdapterMaskProcessor()
                masks = processor.preprocess([BLANK_IMAGE], height=IMAGE_SIZE, width=IMAGE_SIZE)
                masks={"ip_adapter_masks": masks}
            if image is None:
                res = self._text2img(prompt, edges, negative_prompt, controlnet_scale, ip_adapter_image, ip_adapter_scale, masks)
            else:
                res = self._img2img(prompt, edges, negative_prompt, image, strength, controlnet_scale, ip_adapter_image, ip_adapter_scale, masks)
            
            if fix_edges:
                return self.fix_edges(res, edges)
            return res
    
    def fix_edges(self, image, edges):
        with torch.no_grad():
            return edge_registration(self, image, edges)

    
