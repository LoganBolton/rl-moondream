import safetensors
import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Callable

@contextmanager
def safetensors_open(safetensors_file: str):
    """
    Simplify interfacing with safetensors files.
    """
    with safetensors.safe_open(safetensors_file, framework="pt") as st:
        def get_tensor(name: str) -> torch.Tensor:
            return st.get_tensor(name)
        
        def get_keys():
            return st.keys()
        
        get_tensor.keys = get_keys
        yield get_tensor

def simple_load_weights(weights_file: str, model: nn.Module) -> None:
    """
    Load weights from a safetensors file into a MoondreamModel instance.
    This matches the actual structure of the weights file.
    """
    with safetensors_open(weights_file) as get_tensor:
        model = model.to(dtype=torch.float16)
        
        # Vision Model
        print("Loading vision weights...")
        model.vision["patch_emb"].weight.data.copy_(get_tensor("model.vision.patch_emb.weight"))
        model.vision["patch_emb"].bias.data.copy_(get_tensor("model.vision.patch_emb.bias"))
        model.vision.pos_emb.data.copy_(get_tensor("model.vision.pos_emb"))
        
        # Vision blocks
        for i in range(len(model.vision["blocks"])):
            prefix = f"model.vision.blocks.{i}"
            
            # Layer norms
            model.vision["blocks"][i]["ln1"].weight.data.copy_(get_tensor(f"{prefix}.ln1.weight"))
            model.vision["blocks"][i]["ln1"].bias.data.copy_(get_tensor(f"{prefix}.ln1.bias"))
            model.vision["blocks"][i]["ln2"].weight.data.copy_(get_tensor(f"{prefix}.ln2.weight"))
            model.vision["blocks"][i]["ln2"].bias.data.copy_(get_tensor(f"{prefix}.ln2.bias"))
            
            # Attention
            model.vision["blocks"][i]["attn"]["qkv"].weight.data.copy_(get_tensor(f"{prefix}.attn.qkv.weight"))
            model.vision["blocks"][i]["attn"]["qkv"].bias.data.copy_(get_tensor(f"{prefix}.attn.qkv.bias"))
            model.vision["blocks"][i]["attn"]["proj"].weight.data.copy_(get_tensor(f"{prefix}.attn.proj.weight"))
            model.vision["blocks"][i]["attn"]["proj"].bias.data.copy_(get_tensor(f"{prefix}.attn.proj.bias"))
            
            # MLP
            model.vision["blocks"][i]["mlp"]["fc1"].weight.data.copy_(get_tensor(f"{prefix}.mlp.fc1.weight"))
            model.vision["blocks"][i]["mlp"]["fc1"].bias.data.copy_(get_tensor(f"{prefix}.mlp.fc1.bias"))
            model.vision["blocks"][i]["mlp"]["fc2"].weight.data.copy_(get_tensor(f"{prefix}.mlp.fc2.weight"))
            model.vision["blocks"][i]["mlp"]["fc2"].bias.data.copy_(get_tensor(f"{prefix}.mlp.fc2.bias"))
        
        # Vision post processing
        model.vision["post_ln"].weight.data.copy_(get_tensor("model.vision.post_ln.weight"))
        model.vision["post_ln"].bias.data.copy_(get_tensor("model.vision.post_ln.bias"))
        
        model.vision["proj_mlp"]["fc1"].weight.data.copy_(get_tensor("model.vision.proj_mlp.fc1.weight"))
        model.vision["proj_mlp"]["fc1"].bias.data.copy_(get_tensor("model.vision.proj_mlp.fc1.bias"))
        model.vision["proj_mlp"]["fc2"].weight.data.copy_(get_tensor("model.vision.proj_mlp.fc2.weight"))
        model.vision["proj_mlp"]["fc2"].bias.data.copy_(get_tensor("model.vision.proj_mlp.fc2.bias"))
        
        # Text Model
        print("Loading text weights...")
        model.text.wte.data.copy_(get_tensor("model.text.wte"))
        
        # Text blocks
        for i in range(len(model.text["blocks"])):
            prefix = f"model.text.blocks.{i}"
            
            # Layer norm
            model.text["blocks"][i]["ln"].weight.data.copy_(get_tensor(f"{prefix}.ln.weight"))
            model.text["blocks"][i]["ln"].bias.data.copy_(get_tensor(f"{prefix}.ln.bias"))
            
            # Attention
            model.text["blocks"][i]["attn"]["qkv"].weight.data.copy_(get_tensor(f"{prefix}.attn.qkv.weight"))
            model.text["blocks"][i]["attn"]["qkv"].bias.data.copy_(get_tensor(f"{prefix}.attn.qkv.bias"))
            model.text["blocks"][i]["attn"]["proj"].weight.data.copy_(get_tensor(f"{prefix}.attn.proj.weight"))
            model.text["blocks"][i]["attn"]["proj"].bias.data.copy_(get_tensor(f"{prefix}.attn.proj.bias"))
            
            # MLP
            model.text["blocks"][i]["mlp"]["fc1"].weight.data.copy_(get_tensor(f"{prefix}.mlp.fc1.weight"))
            model.text["blocks"][i]["mlp"]["fc1"].bias.data.copy_(get_tensor(f"{prefix}.mlp.fc1.bias"))
            model.text["blocks"][i]["mlp"]["fc2"].weight.data.copy_(get_tensor(f"{prefix}.mlp.fc2.weight"))
            model.text["blocks"][i]["mlp"]["fc2"].bias.data.copy_(get_tensor(f"{prefix}.mlp.fc2.bias"))
        
        # Text post processing
        model.text["post_ln"].weight.data.copy_(get_tensor("model.text.post_ln.weight"))
        model.text["post_ln"].bias.data.copy_(get_tensor("model.text.post_ln.bias"))
        
        model.text["lm_head"].weight.data.copy_(get_tensor("model.text.lm_head.weight"))
        model.text["lm_head"].bias.data.copy_(get_tensor("model.text.lm_head.bias"))
        
        # Region Model
        print("Loading region weights...")
        model.region.coord_features.data.copy_(get_tensor("model.region.coord_features"))
        model.region["coord_encoder"].weight.data.copy_(get_tensor("model.region.coord_encoder.weight"))
        model.region["coord_encoder"].bias.data.copy_(get_tensor("model.region.coord_encoder.bias"))
        
        model.region["coord_decoder"]["fc1"].weight.data.copy_(get_tensor("model.region.coord_decoder.fc1.weight"))
        model.region["coord_decoder"]["fc1"].bias.data.copy_(get_tensor("model.region.coord_decoder.fc1.bias"))
        model.region["coord_decoder"]["fc2"].weight.data.copy_(get_tensor("model.region.coord_decoder.fc2.weight"))
        model.region["coord_decoder"]["fc2"].bias.data.copy_(get_tensor("model.region.coord_decoder.fc2.bias"))
        
        model.region.size_features.data.copy_(get_tensor("model.region.size_features"))
        model.region["size_encoder"].weight.data.copy_(get_tensor("model.region.size_encoder.weight"))
        model.region["size_encoder"].bias.data.copy_(get_tensor("model.region.size_encoder.bias"))
        
        model.region["size_decoder"]["fc1"].weight.data.copy_(get_tensor("model.region.size_decoder.fc1.weight"))
        model.region["size_decoder"]["fc1"].bias.data.copy_(get_tensor("model.region.size_decoder.fc1.bias"))
        model.region["size_decoder"]["fc2"].weight.data.copy_(get_tensor("model.region.size_decoder.fc2.weight"))
        model.region["size_decoder"]["fc2"].bias.data.copy_(get_tensor("model.region.size_decoder.fc2.bias"))
        
        print("Weight loading complete!")
        
        # Make all parameters contiguous
        for param in model.parameters():
            param.data = param.data.contiguous() 