import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import loralib as lora
from timm import create_model
import os
import math
from PIL import Image
from tokenizer import BPETokenizer  
from decoder import Decoder, Config  
import matplotlib.pyplot as plt
import cv2

# 設定路徑
TEST_IMAGES_DIR = "./hw3_data/p3_data/images/"
OUTPUT_DIR = "./visualizations/"
MODEL_CHECKPOINT = "./peft_model_2.0.pth"
DECODER_PATH = "./hw3_data/p2_data/decoder_model.bin"
TOKENIZER_PATH = "."
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    
def load_model_for_inference(model, load_path="peft_model_2.0.pth"):
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint, strict=False)
        print(f"Model parameters loaded from {load_path}")
    else:
        print(f"No checkpoint found at {load_path}")

class VisionEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        self.vit = create_model(model_name, pretrained=pretrained)
        self.vit.reset_classifier(0)
        self.feature_dim = self.vit.embed_dim  # Typically 768 for ViT-Base
        self.embedding_dim = self.feature_dim
        self.feature_resize = nn.Linear(self.feature_dim, self.embedding_dim)
        for name, param in self.vit.named_parameters():
            if 'blocks.11' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for param in self.feature_resize.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.vit.forward_features(x)  # Extract features from ViT
        features = features[:, 1:, :]  # Exclude [CLS] token
        features = self.feature_resize(features)  # Project features
        return features  # Shape: [batch_size, num_patches, embed_dim]

class LoRADecoder(nn.Module):
    def __init__(self, decoder_model_path, rank=4, alpha=8):
        super().__init__()
        cfg = Config(checkpoint=decoder_model_path)
        self.decoder = Decoder(cfg)
        self.embedding_dim = cfg.n_embd
        self.apply_lora_to_decoder(rank, alpha)
        self.token_embedding = self.decoder.transformer.wte
        self.positional_encoding = self.decoder.transformer.wpe
        self.visual_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.visual_projection.requires_grad_(True)
        nn.init.normal_(self.visual_projection.weight, mean=0.0, std=0.02)

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

    def forward(self, vision_features, text_tokens, return_attention=False):
        batch_size, seq_len = text_tokens.size()
        device = text_tokens.device

        # Number of visual tokens
        num_visual_tokens = vision_features.size(1)

        # Text embeddings
        text_embeddings = self.token_embedding(text_tokens)  # [batch_size, seq_len, embed_dim]

        # Visual embeddings
        vision_embeddings = self.visual_projection(vision_features)  # [batch_size, num_visual_tokens, embed_dim]

        # Combine visual and text embeddings
        embeddings = torch.cat([vision_embeddings, text_embeddings], dim=1)  # [batch_size, total_seq_len, embed_dim]

        # Positional embeddings
        total_seq_len = num_visual_tokens + seq_len
        pos_ids = torch.arange(total_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.positional_encoding(pos_ids)

        # Final embeddings
        embeddings = embeddings + position_embeddings

        # Create attention mask
        attention_mask = torch.ones(batch_size, total_seq_len, total_seq_len, device=device)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        attention_mask[:, num_visual_tokens:, num_visual_tokens:] = causal_mask
        attention_mask[:, num_visual_tokens:, :num_visual_tokens] = 1  # Allow text tokens to attend to visual tokens

        # Expand attention mask for multiple heads
        n_head = self.decoder.cfg.n_head  # Changed from config to cfg
        attention_mask = attention_mask.unsqueeze(1).expand(-1, n_head, -1, -1)

        attentions = []
        x = embeddings
        for block in self.decoder.transformer.h:
            ln_x = block.ln_1(x)
            B, T, C = ln_x.size()
            att = block.attn
            q, k, v = att.c_attn(ln_x).split(att.n_embd, dim=2)
            k = k.view(B, T, att.n_head, C // att.n_head).transpose(1, 2)
            q = q.view(B, T, att.n_head, C // att.n_head).transpose(1, 2)
            v = v.view(B, T, att.n_head, C // att.n_head).transpose(1, 2)
            attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attention = attention.masked_fill(attention_mask == 0, float('-inf'))
            attention = F.softmax(attention, dim=-1)

            if return_attention:
                attentions.append(attention)

            att_out = att.c_proj((attention @ v).transpose(1, 2).contiguous().view(B, T, C))
            x = x + att_out
            x = x + block.mlp(block.ln_2(x))

        logits = self.decoder.lm_head(self.decoder.transformer.ln_f(x))

        if return_attention:
            # Return logits for text tokens only
            return {'logits': logits[:, num_visual_tokens:, :], 'attentions': attentions}
        return {'logits': logits[:, num_visual_tokens:, :]}

# 初始化模型與 tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BPETokenizer("encoder.json", "vocab.bpe")
vision_encoder = VisionEncoder()
decoder = LoRADecoder(DECODER_PATH, rank=8, alpha=16)
model = ImageCaptioningModel(vision_encoder, decoder).to(device)
load_model_for_inference(model, "peft_model_2.0.pth")
model.eval()

# 定義圖像轉換
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 可視化attention map並保存
def visualize_attention(image, caption, attentions, num_visual_tokens, output_path):
    image_np = np.array(image.resize((224, 224)))  # Resize image to match attention map dimensions
    h, w, _ = image_np.shape

    grid_size = int(math.sqrt(num_visual_tokens))

    # 設置子圖列數
    num_tokens = len(caption) + 1
    fig, axes = plt.subplots(1, num_tokens, figsize=(4 * num_tokens, 4))

    # 原始影像
    axes[0].imshow(image_np)
    axes[0].axis("off")
    axes[0].set_title("Original Image", fontsize=12)

    # 將每個 Attention Map 疊加到原始圖像
    for i, (word, attention) in enumerate(zip(caption, attentions)):
        attention_np = attention.detach().cpu().numpy()  # [num_visual_tokens]
        attention_map = attention_np.reshape(grid_size, grid_size)
        attention_map = cv2.resize(attention_map, (w, h))

        # Normalize attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-5)

        # Apply a colormap for better visualization
        heatmap = cv2.applyColorMap((attention_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

        axes[i + 1].imshow(overlay)
        axes[i + 1].axis("off")
        axes[i + 1].set_title(f"Token: {word}", fontsize=10)

    # 保存結果
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def sample_next_token(logits, temperature=0.5, top_k=30, top_p=0.9):
    logits = logits / temperature
    
    # Apply top-k
    topk_logits, topk_indices = torch.topk(logits, top_k)
    
    # Apply top-p (nucleus sampling)
    probs = F.softmax(topk_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Sample from filtered distribution
    filtered_logits = topk_logits.clone()
    filtered_logits[sorted_indices_to_remove] = -float('inf')
    probabilities = F.softmax(filtered_logits, dim=-1)
    next_token = topk_indices[torch.multinomial(probabilities, 1)]
    
    return next_token.item()

# 生成caption並可視化attention
def generate_caption_and_visualize(image_path):
    """
    生成圖像的 Caption 並視覺化 Attention Map。
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        vision_features = model.vision_encoder(image_tensor)
        num_visual_tokens = vision_features.size(1)

        current_tokens = torch.tensor([[tokenizer.encode("<|endoftext|>", allowed_special=["<|endoftext|>"])[0]]], device=device)
        caption = []
        attentions = []

        for _ in range(20):  # 最長生成 15 個 Token
            # 執行 forward，返回 logits 和注意力權重
            outputs = model.decoder(vision_features, current_tokens, return_attention=True)
            logits = outputs['logits']
            attention_weights = outputs['attentions'][-1]  # 提取最後一層的注意力權重

            # 提取 self-attention matrix 的最後一步
            if attention_weights.dim() == 4:
                avg_attention = attention_weights.mean(dim=1)  # [batch_size, total_seq_len, total_seq_len]
                final_attention = avg_attention[0, -1, :]  # [total_seq_len]
                # Extract attention to visual tokens
                final_attention_to_visual = final_attention[:num_visual_tokens]  # [num_visual_tokens]
                attentions.append(final_attention_to_visual)

            # 使用 Top-k 抽樣生成下一個 Token
            next_token = sample_next_token(logits[:, -1, :].squeeze(0), temperature=0.9, top_k=10)
            current_tokens = torch.cat([current_tokens, torch.tensor([[next_token]], device=device)], dim=1)

            word = tokenizer.decode([next_token])
            caption.append(word)

            # 停止條件
            if word == "<|endoftext|>" or (len(caption) > 20 and caption[-1] == caption[-2] == caption[-3]):
                break

        if caption and caption[-1] == "<|endoftext|>":
            caption = caption[:-1]

        filename = os.path.basename(image_path).split('.')[0]
        output_path = os.path.join(OUTPUT_DIR, f"{filename}_visualization.jpg")
        visualize_attention(image, caption, attentions, num_visual_tokens, output_path)
        
        print(f"Caption: {' '.join(caption)}")
        print(f"Saved visualization to {output_path}")

# 處理所有測試圖片
test_images = ["bike.jpg", "girl.jpg", "sheep.jpg", "ski.jpg", "umbrella.jpg"]
for img_name in test_images:
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    generate_caption_and_visualize(img_path)
