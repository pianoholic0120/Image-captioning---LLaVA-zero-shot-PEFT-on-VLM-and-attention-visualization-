import os
import torch
import numpy as np
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def generate_caption_and_attentions(image_path, instruction, generation_config, model, processor):
    system_prompt = """You are an expert image captioner trained to produce clear, focused descriptions that reflect the essential visual content of the image."""
    conversation = [{
        "role": "system",
        "content": system_prompt
    }, {
        "role": "user",
        "content": [
            {"type": "text", "text": instruction},
            {"type": "image"}
        ]
    }]

    raw_image = Image.open(image_path).convert("RGB")
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(
        images=raw_image,
        text=prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    # Generate caption without attentions
    with torch.no_grad():
        generated_outputs = model.generate(
            **inputs,
            max_new_tokens=generation_config["max_new_tokens"],
            do_sample=generation_config["do_sample"],
            num_beams=generation_config["num_beams"],
            temperature=generation_config["temperature"],
            top_p=generation_config["top_p"],
            top_k=generation_config["top_k"],
            min_length=generation_config["min_length"],
            early_stopping=generation_config["early_stopping"],
            return_dict_in_generate=True,
            output_attentions=False,  # We don't need attentions here
        )

    # Decode the generated caption
    generated_ids = generated_outputs.sequences
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)

    # Concatenate input_ids and generated_ids
    input_ids = inputs['input_ids']
    full_input_ids = torch.cat([input_ids, generated_ids[:, input_ids.size(-1):]], dim=-1)

    # Now, run the model's forward method to get attentions
    with torch.no_grad():
        outputs = model(
            input_ids=full_input_ids,
            attention_mask=torch.ones_like(full_input_ids),
            pixel_values=inputs['pixel_values'],
            output_attentions=True,
            use_cache=False,
        )

    # Extract attentions
    attentions = outputs.attentions  # This is a tuple of tensors

    # Stack attentions into a tensor
    attn_tensor = torch.stack(attentions)  # Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
    avg_attn = attn_tensor.mean(dim=(0, 2))  # Average over layers and heads

    # Calculate the number of image tokens
    image_size = processor.image_processor.size
    patch_size = processor.image_processor.patch_size
    num_patches_h = image_size["height"] // patch_size
    num_patches_w = image_size["width"] // patch_size
    num_image_tokens = num_patches_h * num_patches_w

    return caption, avg_attn, raw_image, num_image_tokens

def visualize_attention(image, caption, avg_attn, num_image_tokens, output_path, processor):
    """
    Visualize the self-attention between image tokens and generated caption tokens.
    Concatenate attention maps horizontally and save as a single image.
    """
    if avg_attn is None or avg_attn.numel() == 0:
        print("No attention maps available")
        return

    # Get the size of image patches
    image_size = processor.image_processor.size
    patch_size = processor.image_processor.patch_size
    num_patches_h = image_size["height"] // patch_size
    num_patches_w = image_size["width"] // patch_size

    # Convert image to numpy array
    image = image.resize((image_size["width"], image_size["height"]))
    image_np = np.array(image)

    # Get the total sequence length
    seq_len = avg_attn.shape[1]

    # Calculate the indices for image tokens and text tokens
    num_special_tokens = 1  # Adjust if your model uses more special tokens
    image_token_start = num_special_tokens
    image_token_end = image_token_start + num_image_tokens
    text_token_start = image_token_end

    image_token_indices = range(image_token_start, image_token_end)
    text_token_indices = range(text_token_start, seq_len)

    # Get the indices of the generated tokens
    words = caption.strip().split()
    generated_token_indices = range(seq_len - len(words), seq_len)

    # Initialize a list to store individual attention images
    attention_images = []

    # For each generated token, get attention to image tokens
    for idx, word in enumerate(words):
        token_idx = generated_token_indices[idx]
        # Get attention weights from the current token to image tokens
        attention = avg_attn[0, token_idx, image_token_indices]  # Shape: (num_image_tokens,)

        attention_np = attention.cpu().numpy().astype(np.float32)  # Ensure it's a NumPy array with float32 dtype

        # Reshape attention to 2D grid
        attention_map = attention_np.reshape(num_patches_h, num_patches_w)

        # Normalize attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-5)

        # Resize attention map to image size
        attention_resized = cv2.resize(
            attention_map,
            (image_size["width"], image_size["height"]),
            interpolation=cv2.INTER_LINEAR
        )

        # Apply heatmap
        heatmap = cv2.applyColorMap((attention_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

        # Add word label on the image
        overlay = cv2.putText(
            overlay.copy(),
            f"'{word}'",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Append the overlay image to the list
        attention_images.append(overlay)

    # Concatenate all attention images horizontally
    concatenated_image = cv2.hconcat(attention_images)

    # Save the concatenated image
    cv2.imwrite(output_path, cv2.cvtColor(concatenated_image, cv2.COLOR_RGB2BGR))

    print(f"Saved attention visualization to {output_path}")

# Main program
def main():
    # Model initialization
    model_id = "llava-hf/llava-1.5-7b-hf"  # Replace with the correct model ID
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(model_id)
    processor.image_processor.size = {"height": 224, "width": 224}
    processor.image_processor.patch_size = 14  # Ensure this matches the model's configuration

    generation_config = {
        "max_new_tokens": 30,
        "do_sample": True,
        "num_beams": 3,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 50,
        "min_length": 10,
        "early_stopping": True
    }

    instruction = "Describe the image in detail."

    # Process images
    test_images_dir = "./hw3_data/p3_data/images/"
    output_dir = "./visualizations/"
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(test_images_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        try:
            img_path = os.path.join(test_images_dir, img_name)
            print(f"Processing {img_name}...")

            caption, avg_attn, raw_image, num_image_tokens = generate_caption_and_attentions(
                img_path, instruction, generation_config, model, processor
            )

            output_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_attention.jpg")

            visualize_attention(raw_image, caption, avg_attn, num_image_tokens, output_path, processor)
            print(f"Caption: {caption}")
            print(f"Saved visualization to {output_path}")

        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            continue

if __name__ == "__main__":
    main()