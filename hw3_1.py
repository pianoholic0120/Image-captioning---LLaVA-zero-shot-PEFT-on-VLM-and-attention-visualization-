import os
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
import json
from PIL import Image
from tqdm import tqdm

def clean_caption(caption):
    """Clean the caption by extracting only the relevant description."""
    if "ASSISTANT:" in caption:
        caption = caption.split("ASSISTANT:", 1)[1].strip()
    caption = caption.strip()
    if not caption.endswith('.'):
        caption += '.'
    return caption

def generate_caption(image_path, model, processor, instruction, generation_config):
    system_prompt = """You are an expert image captioner trained to produce clear, focused descriptions that reflect the essential visual content of the image. Follow these guidelines to create captions that align with captioning standards and enhance evaluation metrics:

1. **Identify Key Subjects and Actions:** Clearly describe the main subjects (such as people, animals, or objects) and their actions, focusing on what is most visually significant.
2. **Highlight Necessary Details Only:** Mention specific attributes (like colors, textures, or objects) only when they add to the understanding of the scene.
3. **Minimize Background Information:** Include background details only if they help clarify the main subject or action. Avoid general or redundant phrases.
4. **Use Simple, Direct Language:** Keep the caption concise, avoiding any filler words or redundant phrases. Focus on delivering a straightforward description.
5. **Align with Evaluation Standards:** Ensure that each caption is concise, captures the essence of the image, and avoids unnecessary detail for optimal performance in evaluations."""

    conversation = [{
        "role": "system",
        "content": system_prompt
    }, {
        "role": "user",
        "content": [
            {"type": "text", "text": f"{instruction} Keep the description concise, focused on key subjects and actions, and avoid unnecessary details."}, 
            {"type": "image"}
        ]
    }]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(
        images=raw_image, 
        text=prompt, 
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device, torch.float16)
    
    outputs = model.generate(**inputs, **generation_config)
    captions = []
    for output in outputs:
        caption = processor.decode(output[2:], skip_special_tokens=True)
        caption = clean_caption(caption)
        captions.append(caption)
    
    caption = max(captions, key=lambda x: len(set(x.split())))
    
    if caption:
        caption = caption[0].upper() + caption[1:]
    
    return caption

def process_images(image_folder, output_file):
    model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(model_id)
    processor.image_processor.size = {"height": 336, "width": 336}
    processor.patch_size = 14
    processor.vision_feature_select_strategy = "default"
    generation_config = {
        "max_new_tokens": 30,
        "do_sample": True,
        "num_beams": 5,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 50,
        "min_length": 10,
        "num_return_sequences": 3,
        "early_stopping": True
    }
    
    instruction = "Describe this image using one or more simple sentences."
    captions = {}
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in tqdm(image_files, desc="Processing Images"):
        img_path = os.path.join(image_folder, img_file)
        try:
            filename_without_ext = os.path.splitext(img_file)[0]
            caption = generate_caption(img_path, model, processor, instruction, generation_config)
            captions[filename_without_ext] = caption
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
    
    with open(output_file, 'w') as f:
        json.dump(captions, f)

if __name__ == "__main__":
    import sys
    image_folder = sys.argv[1]
    output_file = sys.argv[2]
    process_images(image_folder, output_file)
