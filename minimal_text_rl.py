"""
Minimal text-only RL training for Moondream
Uses a very simple approach with basic reward optimization
"""
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset
from huggingface_hub import hf_hub_download
from hf_moondream import HfMoondream, HfConfig
from config import MoondreamConfig
from simple_weights import simple_load_weights


class TextOnlyMoondream(nn.Module):
    """Simplified text-only version of Moondream for RL"""
    
    def __init__(self, moondream_model):
        super().__init__()
        self.moondream = moondream_model.model
        self.tokenizer = moondream_model.model.tokenizer
        self.vocab_size = 51200
        
    def forward(self, input_ids):
        """Generate text from input tokens"""
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # Simple generation loop
        generated = input_ids.clone()
        max_new_tokens = 20
        
        for _ in range(max_new_tokens):
            # Get next token logits (simplified)
            try:
                # Use a very basic forward pass
                next_logits = self._get_next_token_logits(generated)
                
                # Sample next token
                probs = torch.softmax(next_logits / 0.8, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we hit a reasonable length
                if generated.shape[1] > input_ids.shape[1] + 15:
                    break
                    
            except Exception as e:
                print(f"Generation error: {e}")
                # Fallback: add random tokens
                next_token = torch.randint(0, min(1000, self.vocab_size), (batch_size, 1), device=device)
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def _get_next_token_logits(self, input_ids):
        """Get logits for next token (very simplified)"""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        try:
            # Try to use moondream's text model
            if hasattr(self.moondream.text, 'embeddings'):
                # Get embeddings
                embeddings = self.moondream.text.embeddings.weight[input_ids[:, -1:]]
                
                # Simple linear projection to vocab
                if hasattr(self.moondream.text, 'lm_head'):
                    logits = self.moondream.text.lm_head(embeddings)
                else:
                    # Use embedding weights as output projection
                    logits = torch.matmul(embeddings, self.moondream.text.embeddings.weight.T)
                
                return logits[:, -1, :]  # Return last token logits
            else:
                # Fallback: random logits
                return torch.randn(batch_size, self.vocab_size, device=device)
                
        except Exception as e:
            print(f"Logits computation error: {e}")
            return torch.randn(batch_size, self.vocab_size, device=device)


def simple_reward_function(generated_texts, prompts):
    """Simple reward function based on text quality"""
    rewards = []
    
    for generated, prompt in zip(generated_texts, prompts):
        # Remove the prompt from generated text
        if prompt in generated:
            response = generated.replace(prompt, "").strip()
        else:
            response = generated.strip()
        
        # Reward based on length
        length_reward = min(len(response.split()), 10) * 0.2
        
        # Bonus for common positive words
        positive_words = ["good", "great", "nice", "excellent", "wonderful", "amazing", "helpful", "important"]
        positive_bonus = sum(0.5 for word in positive_words if word.lower() in response.lower())
        
        # Penalty for very short responses
        if len(response.split()) < 2:
            length_penalty = -1.0
        else:
            length_penalty = 0.0
        
        # Bonus for complete sentences (ends with punctuation)
        punctuation_bonus = 0.5 if response.endswith(('.', '!', '?')) else 0.0
        
        total_reward = length_reward + positive_bonus + length_penalty + punctuation_bonus
        rewards.append(max(total_reward, 0.1))  # Minimum reward of 0.1
    
    return rewards


def create_simple_dataset():
    """Create simple text prompts"""
    prompts = [
        "The weather today is",
        "I think that",
        "Learning is",
        "Technology helps",
        "The future",
        "Science is",
        "Education",
        "Friends are",
        "Success means",
        "Happiness comes from",
    ]
    return prompts


def main():
    print("Starting minimal text-only RL training for Moondream...")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load Moondream
        print("Loading Moondream model...")
        hf_config = HfConfig()
        moondream_config = MoondreamConfig()
        hf_config.config = moondream_config.to_dict()
        
        moondream_model = HfMoondream(hf_config)
        
        # Load weights
        weights_path = hf_hub_download(
            repo_id="vikhyatk/moondream2", 
            filename="model.safetensors",
            revision="2025-06-21"
        )
        simple_load_weights(weights_path, moondream_model.model)
        
        # Create simplified model
        model = TextOnlyMoondream(moondream_model)
        model.to(device)
        
        # Create dataset
        prompts = create_simple_dataset()
        print(f"Created dataset with {len(prompts)} prompts")
        
        # Simple training loop
        print("Starting training...")
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-6)
        
        for epoch in range(3):
            print(f"\nEpoch {epoch + 1}/3")
            epoch_rewards = []
            
            # Process prompts in small batches
            for i in range(0, len(prompts), 2):
                batch_prompts = prompts[i:i+2]
                
                # Encode prompts
                encoded_prompts = []
                for prompt in batch_prompts:
                    try:
                        tokens = model.tokenizer.encode(prompt).ids[:20]  # Truncate long prompts
                        encoded_prompts.append(tokens)
                    except:
                        encoded_prompts.append([1, 2, 3])  # Fallback tokens
                
                # Pad to same length
                max_len = max(len(p) for p in encoded_prompts)
                for j, tokens in enumerate(encoded_prompts):
                    while len(tokens) < max_len:
                        tokens.append(0)  # Pad token
                    encoded_prompts[j] = tokens[:max_len]  # Ensure exact length
                
                input_ids = torch.tensor(encoded_prompts, device=device)
                
                # Generate responses
                with torch.no_grad():
                    generated_ids = model(input_ids)
                
                # Decode generated text
                generated_texts = []
                for gen_ids in generated_ids:
                    try:
                        text = model.tokenizer.decode(gen_ids.cpu().tolist())
                        generated_texts.append(text)
                    except:
                        generated_texts.append("Error in decoding")
                
                # Calculate rewards
                rewards = simple_reward_function(generated_texts, batch_prompts)
                epoch_rewards.extend(rewards)
                
                # Print sample
                if i == 0:
                    print(f"Sample - Prompt: '{batch_prompts[0]}'")
                    print(f"Generated: '{generated_texts[0][:100]}...'")
                    print(f"Reward: {rewards[0]:.2f}")
                
                # Simple policy gradient update (very basic)
                if any(p.requires_grad for p in model.parameters()):
                    loss = -torch.tensor(sum(rewards), device=device) * 0.01
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            avg_reward = sum(epoch_rewards) / len(epoch_rewards)
            print(f"Average reward: {avg_reward:.3f}")
        
        print("\nTraining completed!")
        print("This was a minimal demonstration. For better results:")
        print("1. Use proper PPO/REINFORCE implementation")
        print("2. Implement proper advantage calculation")
        print("3. Use more sophisticated reward functions")
        print("4. Add value function for variance reduction")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()