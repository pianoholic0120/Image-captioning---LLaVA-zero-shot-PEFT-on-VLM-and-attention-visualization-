import os
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
import json
from PIL import Image
import subprocess
from tqdm import tqdm

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

generation_configs = [
    {
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
]

instructions = [
    "Describe this image using one or more simple sentences"
]

def clean_caption(caption):
    """Clean the caption by extracting only the relevant description."""
    if "ASSISTANT:" in caption:
        caption = caption.split("ASSISTANT:", 1)[1].strip()
    if caption.endswith('.'):
        caption = caption[:-1]  
    caption = caption.strip()
    
    if any(caption.endswith(word) for word in ['a', 'an', 'the', 'is', 'are', 'being']):
        caption = caption.rsplit(' ', 1)[0]
    
    if not caption.endswith('.'):
        caption = caption + '.'
        
    return caption

def generate_caption(image_path, instruction, generation_config):
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

def optimize_captions(image_folder, output_json="output_llava.json"):
    best_captions = {}
    highest_cider_score = 0
    highest_clip_score = 0
    best_config = None
    
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) 
                  if img.endswith(('.jpg', '.jpeg', '.png'))]
    
    all_scores = []

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    for config in generation_configs:
        for instruction in instructions:
            captions = {}
            for img_path in tqdm(image_paths, desc=f"Processing {instruction[:30]}..."):
                try:
                    filename = os.path.basename(img_path).split('.')[0]
                    caption = generate_caption(img_path, instruction, config.copy())
                    captions[filename] = caption
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
            
            temp_output_file = "temp_output.json"
            try:
                with open(temp_output_file, 'w') as f:
                    json.dump(captions, f)
                
                result = subprocess.run(
                    ["python3", "evaluate.py", 
                     "--pred_file", temp_output_file,
                     "--images_root", "./hw3_data/p1_data/images/val/",
                     "--annotation_file", "./hw3_data/p1_data/val.json"],
                    capture_output=True, 
                    text=True,
                    check=True  
                )
                
                if result.stdout.strip():
                    try:
                        output_parts = result.stdout.strip().split("|")
                        if len(output_parts) >= 2:
                            cider_score = float(output_parts[0].split(":")[-1].strip())
                            clip_score = float(output_parts[1].split(":")[-1].strip())
                        else:
                            print(f"Unexpected output format: {result.stdout}")
                            continue
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing scores: {str(e)}")
                        continue
                else:
                    print("No output from evaluation script")
                    continue
                
                current_config = {
                    "instruction": instruction,
                    "generation_config": config,
                    "cider_score": cider_score,
                    "clip_score": clip_score
                }
                all_scores.append(current_config)
                
                if cider_score > highest_cider_score or \
                   (cider_score == highest_cider_score and clip_score > highest_clip_score):
                    highest_cider_score = cider_score
                    highest_clip_score = clip_score
                    best_captions = captions.copy()
                    best_config = current_config
                
            except subprocess.CalledProcessError as e:
                print(f"Error running evaluation script: {str(e)}")
                print(f"Script output: {e.output}")
                continue
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                continue

    print("\nAll configurations and scores:")
    for score_info in all_scores:
        print(f"Instruction: {score_info['instruction']}")
        print(f"Generation Config: {score_info['generation_config']}")
        print(f"CIDEr Score: {score_info['cider_score']}, CLIP Score: {score_info['clip_score']}\n")

    print("\nBest Configuration:", best_config)
    print("Highest CIDEr Score:", highest_cider_score)
    print("Highest CLIP Score:", highest_clip_score)
    
    with open(output_json, 'w') as f:
        json.dump(best_captions, f)
    
    return highest_cider_score, highest_clip_score

optimize_captions("./hw3_data/p1_data/images/val/")


# CIDEr Score: 1.181254423072826, CLIP Score: 0.7829083251953125