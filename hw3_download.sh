#!/bin/bash

echo "Downloading all required pretrained models using Python..."

# Download ViT-B/16 weights and save state_dict
python -c "
import timm
import torch
import os

# Create model and ensure pretrained weights are downloaded
model = timm.create_model('vit_base_patch16_224', pretrained=True)
os.makedirs('./pretrained_models/vit_base_patch16_224', exist_ok=True)
torch.save(model.state_dict(), './pretrained_models/vit_base_patch16_224/vit_base_patch16_224.pth')
print('ViT-B/16 pretrained weights downloaded and saved.')
"

echo "Model vit_base_patch16_224 downloaded successfully."

wget -O peft_model.pth 'https://www.dropbox.com/scl/fi/qsnpl4a0bzh3yhyiccj3y/peft_model.pth?rlkey=r41jdtqxlaikpawc5zmr8yju6&st=g1g675jm&dl=1'