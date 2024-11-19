import torch
import torch.nn as nn
import json
import os
import re
import random
import numpy as np
from timm import create_model
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tokenizer import BPETokenizer
from decoder import Decoder, Config
from loralib import Linear as LoRALinear
import loralib as lora
from tqdm import tqdm
from torch.cuda.amp import autocast

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1331) 

def load_model_for_inference(model, load_path="peft_model.pth"):
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint, strict=False)
        print(f"Model parameters loaded from {load_path}")
    else:
        print(f"No checkpoint found at {load_path}")

# Vision Encoder
# Vision Encoder
class VisionEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', local_weights_path="pretrained_models/vit_base_patch16_224/vit_base_patch16_224.pth"):
        super().__init__()
        self.vit = create_model(model_name, pretrained=False)  # Initialize model without loading pretrained weights
        if os.path.exists(local_weights_path):
            print(f"Loading pretrained weights from {local_weights_path}...")
            state_dict = torch.load(local_weights_path, map_location=torch.device("cpu"))
            self.vit.load_state_dict(state_dict, strict=False)  # Load pretrained weights
        else:
            raise FileNotFoundError(f"Pretrained weights not found at {local_weights_path}. Please download them first.")

        self.vit.reset_classifier(0)  # Remove classification head
        self.feature_dim = self.vit.embed_dim  # 768 for ViT-Base
        self.embedding_dim = 768
        self.feature_resize = nn.Linear(self.feature_dim, self.embedding_dim)
        for name, param in self.vit.named_parameters():
            if 'blocks.11' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for param in self.feature_resize.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.vit.forward_features(x)  # [B, num_patches + 1, embed_dim]
        return features[:, 0, :]  # Extract [CLS] token

    def forward(self, x):
        features = self.vit.forward_features(x)  # [B, num_patches + 1, embed_dim]
        return features[:, 0, :] 

# LoRA Decoder
class LoRADecoder(nn.Module):
    def __init__(self, decoder_model_path, rank=4, alpha=8):
        super().__init__()
        cfg = Config(checkpoint=decoder_model_path)
        self.decoder = Decoder(cfg)
        self.embedding_dim = cfg.n_embd  
        self.apply_lora_to_decoder(rank, alpha)
        self.token_embedding = self.decoder.transformer.wte
        self.positional_encoding = self.decoder.transformer.wpe
        self.visual_projection = nn.Linear(768, self.embedding_dim)  
        self.visual_projection.requires_grad_(True)
        self.visual_token_embedding = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        self.visual_token_embedding.requires_grad_(True)
        nn.init.normal_(self.visual_token_embedding, mean=0.0, std=0.02)
    def apply_lora_to_decoder(self, rank, alpha):
     for name, module in self.decoder.named_modules():
         if isinstance(module, nn.Linear):
             lora_module = lora.Linear(
                 module.in_features,
                 module.out_features,
                 r=rank,
                 lora_alpha=alpha,
                 lora_dropout=0.05,  
                 merge_weights=False
             )
             parent_module, attr_name = self._get_parent_module_and_attr(name)
             setattr(parent_module, attr_name, lora_module)
     lora.mark_only_lora_as_trainable(self.decoder)

    def _get_parent_module_and_attr(self, module_name):
        attrs = module_name.split(".")
        parent = self.decoder
        for attr in attrs[:-1]:
            parent = getattr(parent, attr)
        return parent, attrs[-1]

    def forward(self, vision_features, text_tokens):
        batch_size, seq_len = text_tokens.size()
        device = text_tokens.device
        text_embeddings = self.token_embedding(text_tokens)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.positional_encoding(pos_ids)
        vision_features = vision_features.to(device)  # [batch_size, vision_dim]
        vision_embeddings = self.visual_projection(vision_features)  # [batch_size, decoder_dim]
        vision_embeddings = vision_embeddings.unsqueeze(1)  # [batch_size, 1, decoder_dim]
        vision_embeddings = vision_embeddings + self.visual_token_embedding
        vision_embeddings = vision_embeddings.expand(-1, seq_len, -1)  # [batch_size, seq_len, decoder_dim]
        embeddings = text_embeddings + position_embeddings + vision_embeddings
        x = embeddings
        for block in self.decoder.transformer.h:
            x = block(x)
        logits = self.decoder.lm_head(x)
        return logits

# Image Captioning Model
class ImageCaptioningModel(nn.Module):
    def __init__(self, vision_encoder, decoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.decoder = decoder

    def forward(self, images, captions, attention_masks):
        vision_features = self.vision_encoder(images)
        logits = self.decoder(vision_features, captions)
        targets = captions[:, 1:]  # Shift target captions
        logits = logits[:, :-1, :]  # Align logits with targets
        attention_masks = attention_masks[:, 1:]  # Align with targets
        return logits, targets, attention_masks

# Post-processing
def post_process_caption(text, min_length=6, max_length=256):
    text = text.replace("<|endoftext|>", "").strip()
    text = re.sub(r'[,\s]+', ' ', text).strip()
    if not text[-1] in {'.', '!', '?'}:
        text += '.'
    text = text.capitalize()
    words = text.split()
    if len(words) < min_length:  
        return text
    articles = {'a', 'an', 'the'}
    connectors = {'and', 'or', 'but', 'of', 'on', 'in', 'at', 'by', 'for', 'with', 'to'}
    keep_words = articles.union(connectors)
    processed_words = []
    prev_word = ""
    for i, word in enumerate(words):
        if (word.lower() != prev_word.lower() or word.lower() in keep_words) and len(processed_words) < max_length:
            if word.lower() in articles:
                if i + 1 < len(words) and words[i + 1].lower() in articles.union(connectors):
                    continue
                if i == len(words) - 1:  
                    continue
            if i == len(words) - 1 and word.lower() in connectors:
                continue
            processed_words.append(word)
            prev_word = word
    processed_text = ' '.join(processed_words)
    if not processed_text[-1] in {'.', '!', '?'}:
        processed_text += '.'
    processed_text = re.sub(r'\s+([.!?])', r'\1', processed_text)
    return processed_text[:max_length]

# Beam Search
def beam_search(model, vision_features, tokenizer, beam_width=3, max_length=14, temperature=0.65, 
                length_penalty=0.9, diversity_penalty=0.5):
    device = next(model.parameters()).device
    start_token_id = tokenizer.encode("<|endoftext|>", allowed_special=["<|endoftext|>"])[0]
    end_token_id = tokenizer.encode("<|endoftext|>", allowed_special=["<|endoftext|>"])[0]
    sequences = [([start_token_id], 0.0)]  
    completed_sequences = []
    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            input_seq = torch.tensor([seq], device=device)
            with torch.no_grad():
                outputs = model.decoder(vision_features, input_seq)
                logits = outputs[:, -1, :]
                logits = logits / temperature
                current_length = len(seq)
                length_factor = ((5 + current_length) ** length_penalty) / ((5 + 1) ** length_penalty)
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_ids = torch.topk(probs, beam_width)
                if len(seq) > 1:
                    for i in range(beam_width):
                        for j in range(i):
                            if topk_ids[0][i] == topk_ids[0][j]:
                                topk_probs[0][i] *= (1 - diversity_penalty)
                for prob, token_id in zip(topk_probs[0], topk_ids[0]):
                    new_seq = seq + [token_id.item()]
                    new_score = score - torch.log(prob).item() / length_factor
                    if token_id.item() == end_token_id:
                        if len(new_seq) >= 5:  
                            completed_sequences.append((new_seq, new_score))
                    else:
                        all_candidates.append((new_seq, new_score))
        if not all_candidates:
            break
        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]
        if len(completed_sequences) >= beam_width * 2:
            break
    if completed_sequences:
        best_seq = sorted(completed_sequences, key=lambda x: x[1])[0][0]
    elif sequences: 
        best_seq = sorted(sequences, key=lambda x: x[1])[0][0]
    else:
        best_seq = [start_token_id]
    return best_seq

# Main Function
def inference(test_image_folder, output_json_path, decoder_weights_path):
    # Initialize tokenizer and model components
    tokenizer = BPETokenizer("encoder.json", "vocab.bpe")
    vision_encoder = VisionEncoder(local_weights_path="pretrained_models/vit_base_patch16_224/vit_base_patch16_224.pth").to(device)
    decoder = LoRADecoder(decoder_weights_path, rank=16, alpha=32).to(device)
    model = ImageCaptioningModel(vision_encoder, decoder).to(device)
    load_model_for_inference(model, "peft_model.pth")
    model.eval()

    # Image Preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Process images
    predictions = {}
    image_list = [f for f in os.listdir(test_image_folder) if f.endswith(('.jpg', '.png'))]
    with torch.no_grad(), autocast():
        with tqdm(total=len(image_list), desc="Processing Images") as pbar:  
            for image_name in image_list:
                image_path = os.path.join(test_image_folder, image_name)
                image = Image.open(image_path).convert("RGB")
                image = transform(image).unsqueeze(0).to(device)
                vision_features = model.vision_encoder(image)
                generated_tokens = beam_search(model, vision_features, tokenizer)
                caption = tokenizer.decode(generated_tokens)
                predictions[os.path.splitext(image_name)[0]] = post_process_caption(caption)
                pbar.update(1)  

    # Save predictions
    with open(output_json_path, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"Predictions saved to {output_json_path}")

if __name__ == "__main__":
    import sys
    test_image_folder = sys.argv[1]
    output_json_path = sys.argv[2]
    decoder_weights_path = sys.argv[3]
    inference(test_image_folder, output_json_path, decoder_weights_path)