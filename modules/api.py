import uvicorn
from fastapi import FastAPI, Body, APIRouter
from pydantic import BaseModel, Field
import json
import io
import base64
from PIL import Image
from typing import List, Union


class TextToImage(BaseModel):
    prompt: str = Field(..., title="Prompt Text", description="The text to generate an image from.")
    negative_prompt: str = Field(default="", title="Negative Prompt Text")
    prompt_style: str = Field(default="None", title="Prompt Style")
    prompt_style2: str = Field(default="None", title="Prompt Style 2")
    steps: int = Field(default=20, title="Steps")
    sampler_index: int = Field(0, title="Sampler Index")
    restore_faces: bool = Field(default=False, title="Restore Faces")
    tiling: bool = Field(default=False, title="Tiling")
    n_iter: int = Field(default=1, title="N Iter")
    batch_size: int = Field(default=1, title="Batch Size")
    cfg_scale: float = Field(default=7, title="Config Scale")
    seed: int = Field(default=-1.0, title="Seed")
    subseed: int = Field(default=-1.0, title="Subseed")
    subseed_strength: float = Field(default=0, title="Subseed Strength")
    seed_resize_from_h: int = Field(default=0, title="Seed Resize From Height")
    seed_resize_from_w: int = Field(default=0, title="Seed Resize From Width")
    height: int = Field(default=512, title="Height")
    width: int = Field(default=512, title="Width")
    enable_hr: bool = Field(default=False, title="Enable HR")
    scale_latent: bool = Field(default=True, title="Scale Latent")
    denoising_strength: float = Field(default=0.7, title="Denoising Strength")


class TextToImageResponse(BaseModel):
    images: List[str] = Field(default=None, title="Image", description="The generated image in base64 format.")
    all_prompts: List[str] = Field(default=None, title="All Prompts", description="The prompt text.")
    negative_prompt: str = Field(default=None, title="Negative Prompt Text")
    seed: int = Field(default=None, title="Seed")
    all_seeds: List[int] = Field(default=None, title="All Seeds")
    subseed: int = Field(default=None, title="Subseed")
    all_subseeds: List[int] = Field(default=None, title="All Subseeds")
    subseed_strength: float = Field(default=None, title="Subseed Strength")
    width: int = Field(default=None, title="Width")
    height: int = Field(default=None, title="Height")
    sampler_index: int = Field(default=None, title="Sampler Index")
    sampler: str = Field(default=None, title="Sampler")
    cfg_scale: float = Field(default=None, title="Config Scale")
    steps: int = Field(default=None, title="Steps")
    batch_size: int = Field(default=None, title="Batch Size")
    restore_faces: bool = Field(default=None, title="Restore Faces")
    face_restoration_model: str = Field(default=None, title="Face Restoration Model")
    sd_model_hash: str = Field(default=None, title="SD Model Hash")
    seed_resize_from_w: int = Field(default=None, title="Seed Resize From Width")
    seed_resize_from_h: int = Field(default=None, title="Seed Resize From Height")
    denoising_strength: float = Field(default=None, title="Denoising Strength")
    extra_generation_params: dict = Field(default={}, title="Extra Generation Params")
    index_of_first_image: int = Field(default=None, title="Index of First Image")
    html: str = Field(default=None, title="HTML")


class ImageToImage(BaseModel):
    prompt: str = Field(..., title="Prompt Text", description="The text to generate an image from.")
    negative_prompt: str = Field(default="", title="Negative Prompt Text")
    prompt_style: str = Field(default="None", title="Prompt Style")
    prompt_style2: str = Field(default="None", title="Prompt Style 2")
    init_img: str = Field(default="", title="init_img")
    init_img_with_mask: None = Field(default=None, title="init_img_with_mask")
    init_mask: None = Field(default=None, title="init_mask")
    mask_mode: None = Field(default=None, title="mask_mode")
    steps: int = Field(default=20, title="Steps")
    sampler_index: int = Field(0, title="Sampler Index")
    mask_blur: int = Field(default=0, title="mask_blur")
    inpainting_fill: int = Field(default=0, title="inpainting_fill")
    restore_faces: bool = Field(default=False, title="Restore Faces")
    tiling: bool = Field(default=False, title="Tiling")
    mode: int = Field(default=0, title="mode")
    n_iter: int = Field(default=1, title="N Iter")
    batch_size: int = Field(default=1, title="Batch Size")
    cfg_scale: float = Field(default=7, title="Config Scale")
    denoising_strength: float = Field(default=0.7, title="Denoising Strength")
    seed: int = Field(default=-1.0, title="Seed")
    subseed: int = Field(default=-1.0, title="Subseed")
    subseed_strength: float = Field(default=-1.0, title="subseed_strength")
    seed_resize_from_h: int = Field(default=0, title="Seed Resize From Height")
    seed_resize_from_w: int = Field(default=0, title="Seed Resize From Width")
    height: int = Field(default=512, title="Height")
    width: int = Field(default=512, title="Width")
    resize_mode: int = Field(default=0, title="resize_mode")
    upscaler_index: str = Field(default=0, title="upscaler_index")
    upscale_overlap: int = Field(default=0, title="upscale_overlap")
    inpaint_full_res: bool = Field(default=False, title="inpaint_full_res")
    inpainting_mask_invert: int = Field(default=0, title="inpainting_mask_invert")


class ImageToImageResponse(BaseModel):
    images: List[str] = Field(default=None, title="Image", description="The generated image in base64 format.")
    all_prompts: List[str] = Field(default=None, title="All Prompts", description="The prompt text.")
    negative_prompt: str = Field(default=None, title="Negative Prompt Text")
    seed: int = Field(default=None, title="Seed")
    all_seeds: List[int] = Field(default=None, title="All Seeds")
    subseed: int = Field(default=None, title="Subseed")
    all_subseeds: List[int] = Field(default=None, title="All Subseeds")
    subseed_strength: float = Field(default=None, title="Subseed Strength")
    width: int = Field(default=None, title="Width")
    height: int = Field(default=None, title="Height")
    sampler_index: int = Field(default=None, title="Sampler Index")
    sampler: str = Field(default=None, title="Sampler")
    cfg_scale: float = Field(default=None, title="Config Scale")
    steps: int = Field(default=None, title="Steps")
    batch_size: int = Field(default=None, title="Batch Size")
    restore_faces: bool = Field(default=None, title="Restore Faces")
    face_restoration_model: str = Field(default=None, title="Face Restoration Model")
    sd_model_hash: str = Field(default=None, title="SD Model Hash")
    seed_resize_from_w: int = Field(default=None, title="Seed Resize From Width")
    seed_resize_from_h: int = Field(default=None, title="Seed Resize From Height")
    denoising_strength: float = Field(default=None, title="Denoising Strength")
    extra_generation_params: dict = Field(default={}, title="Extra Generation Params")
    index_of_first_image: int = Field(default=None, title="Index of First Image")
    infotexts: List[str] = Field(default=None, title="infotexts", description="The infotexts.")
    html: str = Field(default=None, title="HTML")


app = FastAPI()


class Api:
    def __init__(self, txt2img, img2img, run_extras, run_pnginfo):
        self.txt2img = txt2img
        self.img2img = img2img
        self.run_extras = run_extras
        self.run_pnginfo = run_pnginfo

        self.router = APIRouter()
        app.add_api_route("/v1/txt2img", self.txt2imgendoint, response_model=TextToImageResponse)
        app.add_api_route("/v1/img2img", self.img2imgendoint, response_model=ImageToImageResponse)
        app.add_api_route("/v1/extras", self.extrasendoint)
        app.add_api_route("/v1/pnginfo", self.pnginfoendoint)

    def txt2imgendoint(self, txt2imgreq: TextToImage = Body(embed=True)):
        images, params, html = self.txt2img(*list(txt2imgreq.dict().values()), 0, False, None, '', False, 1, '', 4, '', True)

        b64images = []
        for i in images:
            buffer = io.BytesIO()
            i.save(buffer, format="png")
            b64images.append(base64.b64encode(buffer.getvalue()))
        resp_params = json.loads(params)
        return TextToImageResponse(images=b64images, **resp_params, html=html)

    def img2imgendoint(self, img2imgreq: ImageToImage = Body(embed=True)):
        img2imgreq.init_img = Image.open(io.BytesIO(base64.b64decode(img2imgreq.init_img.encode("utf-8"))))
        images, params, html = self.img2img(*list(img2imgreq.dict().values()), 0, False, None, '', False, 1, '', 4, '', True)

        b64images = []
        for i in images:
            buffer = io.BytesIO()
            i.save(buffer, format="png")
            b64images.append(base64.b64encode(buffer.getvalue()))
        resp_params = json.loads(params)
        print(resp_params)
        return ImageToImageResponse(images=b64images, **resp_params, html=html)

    def extrasendoint(self):
        raise NotImplementedError

    def pnginfoendoint(self):
        raise NotImplementedError

    def launch(self, server_name, port):
        app.include_router(self.router)
        uvicorn.run(app, host=server_name, port=port)
