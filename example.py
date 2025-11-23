Hugging Face's logo
Hugging Face
Models
Datasets
Spaces
Community
Docs
Enterprise
Pricing


Spaces:
dream2589632147
/
Dream-wan2-2-faster-Pro


like
232
App
Files
Community
2
Dream-wan2-2-faster-Pro
/
app.py

dream2589632147's picture
dream2589632147
Update app.py
5b9c736
verified
5 days ago
raw

Copy download link
history
blame
contribute
delete

10.2 kB
import os
import spaces
import torch
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils.export_utils import export_to_video
import gradio as gr
import tempfile
import numpy as np
from PIL import Image
import random
import gc

from torchao.quantization import quantize_
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, Int8WeightOnlyConfig
import aoti

# =========================================================
# MODEL CONFIGURATION
# =========================================================
MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"  # المسار الجديد للنموذج
HF_TOKEN = os.environ.get("HF_TOKEN")  # ضع توكن Hugging Face هنا إذا كان النموذج خاصًا

MAX_DIM = 832
MIN_DIM = 480
SQUARE_DIM = 640
MULTIPLE_OF = 16

MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 7720

MIN_DURATION = round(MIN_FRAMES_MODEL / FIXED_FPS, 1)
MAX_DURATION = round(MAX_FRAMES_MODEL / FIXED_FPS, 1)

# =========================================================
# LOAD PIPELINE
# =========================================================
pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    transformer=WanTransformer3DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        token=HF_TOKEN
    ),
    transformer_2=WanTransformer3DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer_2",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        token=HF_TOKEN
    ),
    torch_dtype=torch.bfloat16,
).to("cuda")

# =========================================================
# LOAD LORA ADAPTERS
# =========================================================
pipe.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
    adapter_name="lightx2v"
)
pipe.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
    adapter_name="lightx2v_2",
    load_into_transformer_2=True
)

pipe.set_adapters(["lightx2v", "lightx2v_2"], adapter_weights=[1., 1.])
pipe.fuse_lora(adapter_names=["lightx2v"], lora_scale=3., components=["transformer"])
pipe.fuse_lora(adapter_names=["lightx2v_2"], lora_scale=1., components=["transformer_2"])
pipe.unload_lora_weights()

# =========================================================
# QUANTIZATION & AOT OPTIMIZATION
# =========================================================
quantize_(pipe.text_encoder, Int8WeightOnlyConfig())
quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
quantize_(pipe.transformer_2, Float8DynamicActivationFloat8WeightConfig())

aoti.aoti_blocks_load(pipe.transformer, 'zerogpu-aoti/Wan2', variant='fp8da')
aoti.aoti_blocks_load(pipe.transformer_2, 'zerogpu-aoti/Wan2', variant='fp8da')

# =========================================================
# DEFAULT PROMPTS
# =========================================================
default_prompt_i2v = "make this image come alive, cinematic motion, smooth animation"
default_negative_prompt = (
    "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, "
    "最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, "
    "畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走"
)

# =========================================================
# IMAGE RESIZING LOGIC
# =========================================================
def resize_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image.resize((SQUARE_DIM, SQUARE_DIM), Image.LANCZOS)

    aspect_ratio = width / height
    MAX_ASPECT_RATIO = MAX_DIM / MIN_DIM
    MIN_ASPECT_RATIO = MIN_DIM / MAX_DIM

    image_to_resize = image

    if aspect_ratio > MAX_ASPECT_RATIO:
        crop_width = int(round(height * MAX_ASPECT_RATIO))
        left = (width - crop_width) // 2
        image_to_resize = image.crop((left, 0, left + crop_width, height))
    elif aspect_ratio < MIN_ASPECT_RATIO:
        crop_height = int(round(width / MIN_ASPECT_RATIO))
        top = (height - crop_height) // 2
        image_to_resize = image.crop((0, top, width, top + crop_height))

    if width > height:
        target_w = MAX_DIM
        target_h = int(round(target_w / aspect_ratio))
    else:
        target_h = MAX_DIM
        target_w = int(round(target_h * aspect_ratio))

    final_w = round(target_w / MULTIPLE_OF) * MULTIPLE_OF
    final_h = round(target_h / MULTIPLE_OF) * MULTIPLE_OF

    final_w = max(MIN_DIM, min(MAX_DIM, final_w))
    final_h = max(MIN_DIM, min(MAX_DIM, final_h))

    return image_to_resize.resize((final_w, final_h), Image.LANCZOS)

# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def get_num_frames(duration_seconds: float):
    return 1 + int(np.clip(int(round(duration_seconds * FIXED_FPS)), MIN_FRAMES_MODEL, MAX_FRAMES_MODEL))

def get_duration(
    input_image, prompt, steps, negative_prompt,
    duration_seconds, guidance_scale, guidance_scale_2,
    seed, randomize_seed, progress,
):
    BASE_FRAMES_HEIGHT_WIDTH = 81 * 832 * 624
    BASE_STEP_DURATION = 15
    width, height = resize_image(input_image).size
    frames = get_num_frames(duration_seconds)
    factor = frames * width * height / BASE_FRAMES_HEIGHT_WIDTH
    step_duration = BASE_STEP_DURATION * factor ** 1.5
    return 10 + int(steps) * step_duration

# =========================================================
# MAIN GENERATION FUNCTION
# =========================================================
@spaces.GPU(duration=get_duration)
def generate_video(
    input_image,
    prompt,
    steps=4,
    negative_prompt=default_negative_prompt,
    duration_seconds=MAX_DURATION,
    guidance_scale=1,
    guidance_scale_2=1,
    seed=42,
    randomize_seed=False,
    progress=gr.Progress(track_tqdm=True),
):
    if input_image is None:
        raise gr.Error("Please upload an input image.")

    num_frames = get_num_frames(duration_seconds)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    resized_image = resize_image(input_image)

    output_frames_list = pipe(
        image=resized_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=resized_image.height,
        width=resized_image.width,
        num_frames=num_frames,
        guidance_scale=float(guidance_scale),
        guidance_scale_2=float(guidance_scale_2),
        num_inference_steps=int(steps),
        generator=torch.Generator(device="cuda").manual_seed(current_seed),
    ).frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        video_path = tmpfile.name
    export_to_video(output_frames_list, video_path, fps=FIXED_FPS)
    return video_path, current_seed

# =========================================================
# GRADIO UI
# =========================================================
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Dream Wan 2.2 Faster Pro (14B) — Ultra Fast I2V with Lightning LoRA")
    gr.Markdown("Optimized FP8 quantized pipeline with AoT blocks & 4-step fast inference ⚡")

    with gr.Row():
        with gr.Column():
            input_image_component = gr.Image(type="pil", label="Input Image")
            prompt_input = gr.Textbox(label="Prompt", value=default_prompt_i2v)
            duration_seconds_input = gr.Slider(
                minimum=MIN_DURATION, maximum=MAX_DURATION, step=0.1, value=3.5,
                label="Duration (seconds)",
                info=f"Model range: {MIN_FRAMES_MODEL}-{MAX_FRAMES_MODEL} frames at {FIXED_FPS}fps."
            )

            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt_input = gr.Textbox(label="Negative Prompt", value=default_negative_prompt, lines=3)
                seed_input = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42)
                randomize_seed_checkbox = gr.Checkbox(label="Randomize seed", value=True)
                steps_slider = gr.Slider(minimum=1, maximum=30, step=1, value=6, label="Inference Steps")
                guidance_scale_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale (high noise)")
                guidance_scale_2_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale 2 (low noise)")

            generate_button = gr.Button("🎬 Generate Video", variant="primary")

        with gr.Column():
            video_output = gr.Video(label="Generated Video", autoplay=True)

    ui_inputs = [
        input_image_component, prompt_input, steps_slider,
        negative_prompt_input, duration_seconds_input,
        guidance_scale_input, guidance_scale_2_input,
        seed_input, randomize_seed_checkbox
    ]
    generate_button.click(fn=generate_video, inputs=ui_inputs, outputs=[video_output, seed_input])

    gr.Examples(
        examples=[
            [
                "wan_i2v_input.JPG",
                "POV selfie video, white cat with sunglasses standing on surfboard, relaxed smile, tropical beach behind (clear water, green hills, blue sky with clouds). Surfboard tips, cat falls into ocean, camera plunges underwater with bubbles and sunlight beams. Brief underwater view of cat’s face, then cat resurfaces, still filming selfie, playful summer vacation mood.",
                4,
            ],
        ],
        inputs=[input_image_component, prompt_input, steps_slider],
        outputs=[video_output, seed_input],
        fn=generate_video,
        cache_examples="lazy"
    )

if __name__ == "__main__":
    demo.queue().launch(mcp_server=True)

