"""
PPO-style LoRA-based RL training for Moondream, V19 (Final, Corrected Learning).

This version solves the "zero-reward, zero-loss" cycle by implementing two crucial
changes to the RL training logic, enabling the model to learn effectively.

Key Fixes:
1.  **Smoother Reward Function:** Replaced the punitive reward function with a simple,
    effective linear penalty. The reward is now always non-zero, providing a
    consistent gradient for the model to learn from, even when its output is
    far from the target.
2.  **EOS Token Bias:** A small bonus is added to the logit of the End-of-Sequence
    token during generation. This gently encourages the model to learn to stop,
    helping it break out of the cycle of always hitting the max token limit.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from moondream import MoondreamModel
from config import MoondreamConfig
from text import text_encoder, lm_head, text_decoder
from huggingface_hub import hf_hub_download
# This import assumes you have a 'simple_weights.py' file with this function.
from simple_weights import simple_load_weights 
import random
import wandb
import time
import copy
import os

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

class CleanLoRALayer(nn.Module):
    """Clean LoRA implementation."""
    
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        weight_shape = original_layer.weight.shape
        out_features, in_features = weight_shape[0], weight_shape[1]
        device = original_layer.weight.device
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32, device=device))
        
        nn.init.normal_(self.lora_A, std=1/rank)
        
        self.scaling = self.alpha / self.rank
    
    def forward(self, x):
        lora_result = (x.float() @ self.lora_A.T) @ self.lora_B.T
        return self.original_layer(x) + (lora_result.to(x.dtype) * self.scaling)

class PpoLoRATrainerV19:
    """
    PPO trainer with a corrected reward landscape and EOS bias.
    """
    
    def __init__(self, moondream_model, scaler, target_tokens=50, kl_weight=0.1, max_gen_tokens=70):
        self.active_model = moondream_model
        
        # FIX: Correctly access the underlying model module when DataParallel is used
        model_module = self.active_model.module if isinstance(self.active_model, nn.DataParallel) else self.active_model
        self.tokenizer = model_module.tokenizer
        self.config = model_module.config
        
        self.scaler = scaler
        
        self.target_tokens = target_tokens
        self.kl_weight = kl_weight
        self.max_gen_tokens = max_gen_tokens
        self.device = next(self.active_model.parameters()).device
        self.eos_id = self.config.tokenizer.eos_id
        self.max_context = self.config.text.max_context
        
        print("   Creating frozen reference model (on primary device)...")
        self.ref_model = copy.deepcopy(model_module).to(self.device)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
        
        lora_params = [p for p in self.active_model.parameters() if p.requires_grad]
        print(f"   Correctly found {sum(p.numel() for p in lora_params):,} trainable parameters.")
        self.optimizer = optim.Adam(lora_params, lr=1e-4)

    def _get_logits(self, model, input_ids):
        model_module = model.module if isinstance(model, nn.DataParallel) else model
        model_module._setup_caches()
        text_model = model_module.text

        inputs_embeds = text_encoder(input_ids, text_model)
        seq_len = inputs_embeds.shape[1]
        
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        
        full_causal_mask = torch.tril(
            torch.ones(1, 1, self.max_context, self.max_context, device=input_ids.device, dtype=torch.bool)
        )
        attn_mask = full_causal_mask[:, :, :seq_len, :self.max_context]
        
        hidden_states = text_decoder(
            inputs_embeds, text_model, attn_mask, pos_ids, self.config.text, lora=None
        )
        
        logits = lm_head(hidden_states, text_model)
        return logits

    def compute_reward(self, token_count):
        distance = abs(token_count - self.target_tokens)
        reward = 1.0 - (distance / self.max_gen_tokens)
        return reward
    
    def generate_and_train(self, prompt_text, eos_bias=0.0):
        bos_id = self.config.tokenizer.bos_id
        prompt_tokens = [bos_id] + self.tokenizer.encode(prompt_text).ids
        input_ids = torch.tensor([prompt_tokens], device=self.device)

        generated_tokens, active_log_probs, kl_divs = [], [], []

        # FINAL FIX: Programmatically get the model's dtype and use it for autocast
        # This ensures the training loop's dtype always matches the model's dtype.
        model_dtype = next(self.active_model.parameters()).dtype
        with torch.amp.autocast(device_type='cuda', dtype=model_dtype):
            for _ in range(self.max_gen_tokens):
                active_logits = self._get_logits(self.active_model, input_ids)
                with torch.no_grad():
                    ref_logits = self._get_logits(self.ref_model, input_ids)

                if active_logits.dim() == 3:
                    active_logits_last = active_logits[:, -1, :]
                    ref_logits_last = ref_logits[:, -1, :]
                else:
                    active_logits_last = active_logits[-1, :]
                    ref_logits_last = ref_logits[-1, :]

                # Cast to float32 for softmax and other calculations to maintain precision
                active_logits_last = active_logits_last.float()
                ref_logits_last = ref_logits_last.float()
                
                active_logits_last[..., self.eos_id] += eos_bias

                probs = F.softmax(active_logits_last, dim=-1)
                if probs.dim() == 1:
                    probs = probs.unsqueeze(0)
                
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() == self.eos_id:
                    break
                
                active_log_softmax = F.log_softmax(active_logits_last, dim=-1)
                log_prob = active_log_softmax.squeeze()[next_token.item()]
                active_log_probs.append(log_prob)
                
                kl_div = F.kl_div(
                    active_log_softmax, F.softmax(ref_logits_last, dim=-1), 
                    reduction='batchmean', log_target=False
                )
                kl_divs.append(kl_div)

                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                del active_logits, ref_logits

        if not generated_tokens:
            return 0, 0, 0, ""

        token_count = len(generated_tokens)
        reward = self.compute_reward(token_count)
        
        policy_loss = -reward * torch.stack(active_log_probs).mean()
        kl_penalty = torch.stack(kl_divs).mean()
        loss = policy_loss + self.kl_weight * kl_penalty
        
        self.scaler.scale(loss).backward()
        
        generated_text = self.tokenizer.decode(generated_tokens)
        return loss.item(), reward, token_count, generated_text

    def train_episode(self, prompts, accumulation_steps, eos_bias):
        all_data = []
        total_loss = 0.0
        self.optimizer.zero_grad()

        for i, prompt in enumerate(prompts):
            loss, reward, token_count, generated = self.generate_and_train(prompt, eos_bias)
            if loss == 0 and reward == 0: # Skip updates if nothing was generated
                continue
                
            loss_val = loss / accumulation_steps
            total_loss += loss_val

            all_data.append({
                'reward': reward, 'tokens': token_count, 
                'prompt': prompt, 'generated': generated
            })

            if (i + 1) % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in self.active_model.parameters() if p.requires_grad], max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()

        avg_reward = sum(d['reward'] for d in all_data) / len(all_data) if all_data else 0
        all_token_counts = [d['tokens'] for d in all_data]
        
        return avg_reward, all_token_counts, all_data, total_loss
    
def add_lora_layers(model):
    """Helper function to add LoRA layers to the model."""
    for i, block in enumerate(model.text.blocks):
        if i >= len(model.text.blocks) - 3:
            block.attn.proj = CleanLoRALayer(block.attn.proj, rank=8, alpha=16)
            block.mlp.fc2 = CleanLoRALayer(block.mlp.fc2, rank=8, alpha=16)

    model.text.lm_head = CleanLoRALayer(model.text.lm_head, rank=8, alpha=16)
    return model

def main():
    print("🚀 Starting PPO-style LoRA RL Training for Moondream (V19 - Corrected Learning)")
    
    max_gen_tokens = 150
    batch_size = 1
    accumulation_steps = 1
    eos_bias = 0.5 # A small positive value to encourage EOS
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Primary Device: {device}")
    
    if torch.cuda.is_available():
        print(f"Visible GPUs: {torch.cuda.device_count()}")
        if not torch.cuda.is_bf16_supported():
             print("Warning: bfloat16 is not supported on this device.")
    
    wandb.init(project="moondream-ppo-lora-rl-v19", name=f"ppo-lora-learning-fix-{int(time.time())}")
    
    try:
        print("\n📦 Loading Moondream...")
        config = MoondreamConfig()
        
        # FIX: Intelligently select model dtype based on hardware support
        model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        moondream_model = MoondreamModel(config, dtype=model_dtype)
        
        weights_path = hf_hub_download(repo_id="vikhyatk/moondream2", filename="model.safetensors")
        
        # FIX: Use a dedicated weight loading function to prevent silent failures
        print("   Loading weights into model using simple_load_weights...")
        simple_load_weights(weights_path, moondream_model)
        print("   Model weights loaded successfully!")
        
        print("   Freezing all base model parameters...")
        for param in moondream_model.parameters():
            param.requires_grad = False
        
        print("   Adding LoRA layers...")
        moondream_model = add_lora_layers(moondream_model)
        
        moondream_model = moondream_model.to(device)
        if torch.cuda.device_count() > 1:
            print(f"   Using {torch.cuda.device_count()} GPUs via DataParallel.")
            moondream_model = nn.DataParallel(moondream_model)
        
        scaler = torch.amp.GradScaler()
        
        print("   Setting up PPO LoRA trainer...")
        rl_trainer = PpoLoRATrainerV19(moondream_model, scaler, target_tokens=50, kl_weight=0.1, max_gen_tokens=max_gen_tokens)
        
        train_prompts = [
            "The weather today is", "I believe that", "Technology helps us", "Learning is important because", 
            "In the future", "Science shows us", "Education means", "Success comes from", "Friendship is", "The best way to"
        ]
        
        num_episodes = 200
        total_batch_size = batch_size * accumulation_steps
        print(f"\n🏃 Training for {num_episodes} episodes with total batch size {total_batch_size}...")
        wandb.config.update({"num_episodes": num_episodes, "max_gen_tokens": max_gen_tokens, "batch_size": batch_size, "accumulation_steps": accumulation_steps, "eos_bias": eos_bias})
        
        for episode in range(num_episodes):
            batch_prompts = random.sample(train_prompts, total_batch_size)
            
            avg_reward, token_counts, episode_data, loss = rl_trainer.train_episode(batch_prompts, accumulation_steps, eos_bias)
            
            avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
            
            wandb.log({"episode": episode + 1, "avg_reward": avg_reward, "avg_tokens": avg_tokens, "loss": loss})
            print(f"E {episode+1:3d}: Reward={avg_reward:6.2f}, Tokens={avg_tokens:6.1f}, Loss={loss:7.4f}")
            
        print("\n🎉 LoRA RL Training completed!")
        wandb.finish()
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        if wandb.run and wandb.run.id:
            wandb.finish(exit_code=1)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()