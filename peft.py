import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import loralib as lora
from timm import create_model
import json
from tqdm import tqdm
import os
import re
import math
import random
from PIL import Image
from tokenizer import BPETokenizer  
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from decoder import Decoder, Config  
from evaluate import CIDERScore, CLIPScore, getGTCaptions, readJSON
from collections import defaultdict
from torch.amp import autocast, GradScaler

DECODER_PATH = './hw3_data/p2_data/decoder_model.bin'
TOKENIZER_PATH = '.'
TRAIN_DATA_PATH = './hw3_data/p2_data/images/train/'
VAL_DATA_PATH = './hw3_data/p2_data/images/val/'
VAL_JSON_PATH = './hw3_data/p2_data/val.json'

def save_best_model(model, save_path="peft_model.pth"):
    torch.save(model.state_dict(), save_path)  
    print(f"Model parameters saved to {save_path}")

def load_model_for_inference(model, load_path="peft_model.pth"):
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint, strict=False)
        print(f"Model parameters loaded from {load_path}")
    else:
        print(f"No checkpoint found at {load_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()

def collate_fn(batch):
    max_length = 50
    images, captions, filenames = zip(*batch)
    images = torch.stack(images)
    pad_token_id = tokenizer.encode("<|endoftext|>", allowed_special=["<|endoftext|>"])[0]
    captions = [cap[:max_length] for cap in captions]
    captions = pad_sequence(captions, batch_first=True, padding_value=pad_token_id)
    attention_masks = (captions != pad_token_id).long()
    return images, captions, attention_masks, filenames
    
class VisionEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        self.vit = create_model(model_name, pretrained=pretrained)
        self.vit.reset_classifier(0)
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
        return features[:, 0, :] 

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

class ImageCaptionDataset(Dataset):
    def __init__(self, img_folder, annotations_file, tokenizer, transform=None):
        self.img_folder = img_folder
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)["annotations"]
        self.tokenizer = tokenizer
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_id = annotation['image_id']
        caption = annotation['caption']
        
        img_filename = f"{img_id:012}.jpg"
        img_path = os.path.join(self.img_folder, img_filename)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        encoded_caption = self.tokenizer.encode(caption, allowed_special=["<|endoftext|>"])
        start_token = self.tokenizer.encode("<|endoftext|>", allowed_special=["<|endoftext|>"])
        end_token = self.tokenizer.encode("<|endoftext|>", allowed_special=["<|endoftext|>"])
        tokenized_caption = torch.tensor(start_token + encoded_caption + end_token, dtype=torch.long)

        return image, tokenized_caption, img_filename

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def create_fixed_eval_subset(dataset, subset_size=15):
    subset_indices = random.sample(range(len(dataset)), subset_size)
    subset = torch.utils.data.Subset(dataset, subset_indices)
    return DataLoader(subset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

def load_val_captions(val_json_path):
    with open(val_json_path, 'r') as f:
        data = json.load(f)
    val_captions = {str(item["image_id"]).zfill(12): item["caption"] for item in data["annotations"]}
    return val_captions

tokenizer = BPETokenizer("encoder.json", "vocab.bpe")
train_dataset = ImageCaptionDataset(img_folder=TRAIN_DATA_PATH, annotations_file='./hw3_data/p2_data/train.json', tokenizer=tokenizer, transform=transform)
val_dataset = ImageCaptionDataset(img_folder=VAL_DATA_PATH, annotations_file=VAL_JSON_PATH, tokenizer=tokenizer, transform=val_transform)
val_captions = load_val_captions(VAL_JSON_PATH)  
eval_subset_loader = create_fixed_eval_subset(val_dataset, subset_size=5)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_fn)

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

# Initialize model components
vision_encoder = VisionEncoder()
for param in vision_encoder.parameters():
    param.requires_grad = False
decoder = LoRADecoder(DECODER_PATH, rank=16, alpha=32) # Using the updated LoRADecoder with Config and Decoder
model = ImageCaptioningModel(vision_encoder, decoder).to(device)
load_model_for_inference(model, "peft_model.pth")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {trainable_params / 1e6:.2f}M")

def compute_loss(logits, targets, attention_masks):
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    loss = loss.view(targets.size(0), -1)
    loss = loss * attention_masks
    return loss.sum() / attention_masks.sum()

def train(model, train_loader, val_loader, tokenizer, epochs=10):
    torch.cuda.empty_cache()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    pad_token_id = tokenizer.encode("<|endoftext|>", allowed_special=["<|endoftext|>"])[0]
    # criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1, reduction='mean')
    early_stopping_patience = 20
    best_loss = float('inf')
    patience_counter = 0
    max_grad_norm = 0.5
    scaler = GradScaler()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, captions, attention_masks, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images, captions, attention_masks = images.to(device), captions.to(device), attention_masks.to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                logits, targets, masks = model(images, captions, attention_masks)
                loss = compute_loss(logits, targets, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

        model.eval()
        subset_predictions = {}
        with torch.no_grad():
            for images, captions, attention_masks, filenames in tqdm(eval_subset_loader, desc="Evaluating Subset"):
                images = images.to(device)
                for idx, image in enumerate(images):
                    image = image.unsqueeze(0)
                    generated_tokens = beam_search(model, image, tokenizer, beam_width=5, max_length=50, temperature=0.8)
                    predicted_text = tokenizer.decode(generated_tokens)
                    predicted_text = post_process_caption(predicted_text)
                    filename = os.path.splitext(filenames[idx])[0]
                    subset_predictions[filename] = predicted_text
        subset_gts = {img_id: val_captions[img_id] for img_id in subset_predictions.keys()}
        cider_score = CIDERScore()(subset_predictions, subset_gts)
        clip_score = CLIPScore()(subset_predictions, VAL_DATA_PATH)
        print(f"Epoch {epoch + 1} - CIDEr Score on Subset: {cider_score:.4f} | CLIPScore on Subset: {clip_score:.4f}")
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            save_best_model(model, save_path="peft_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
    evaluate(model, val_loader, tokenizer)

def beam_search(model, vision_features, tokenizer, beam_width=6, max_length=25, temperature=0.8, 
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
            with torch.no_grad(), autocast(device_type='cuda'):
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

def evaluate(model, val_loader, tokenizer, annotation_file=VAL_JSON_PATH, images_root=VAL_DATA_PATH, output_file="predictions.json"):
    model.eval()
    predictions = {}
    clip_score_calculator = CLIPScore()
    with torch.no_grad(), autocast(device_type="cuda"):
        for images, captions, attention_masks, filenames in tqdm(val_loader):
            images = images.to(device)
            vision_features = model.vision_encoder(images)
            # print(f"Vision Features Shape: {vision_features.shape}")
            for idx, filename in enumerate(filenames):
                filename = os.path.splitext(filename)[0]
                if filename in predictions:
                    continue
                vision_feature = vision_features[idx].unsqueeze(0)
                generated_tokens = beam_search(model, vision_feature, tokenizer, beam_width=3, max_length=14, temperature=0.65)
                predicted_text = tokenizer.decode(generated_tokens)
                predicted_text = post_process_caption(predicted_text)
                if not predicted_text.strip():
                    predicted_text = "An image."
                predictions[filename] = predicted_text
                clip_score = clip_score_calculator({filename: predicted_text}, images_root)
                print(f"Predicted text for {filename}:", predicted_text, f"CLIP_SCORE: {clip_score}")
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"Predictions saved to {output_file}")
    annotations = readJSON(annotation_file)
    gts = getGTCaptions(annotations)
    cider_score = CIDERScore()(predictions, gts)
    clip_score = CLIPScore()(predictions, images_root)
    print(f"CIDEr: {cider_score:.4f} | CLIPScore: {clip_score:.4f}")

# train(model, train_loader, val_loader, tokenizer, epochs=100)
evaluate(model, val_loader, tokenizer)