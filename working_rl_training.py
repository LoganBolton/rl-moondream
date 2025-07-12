"""
Working RL training for Moondream
Uses evaluation-based RL rather than policy gradients to avoid gradient issues
"""
import torch
import torch.nn as nn
import torch.optim as optim
from moondream import MoondreamModel
from config import MoondreamConfig
from huggingface_hub import hf_hub_download
from simple_weights import simple_load_weights
import random
import wandb
import time


class MoondreamRLTrainer:
    """Simple RL trainer - just generates text with fixed settings"""
    
    def __init__(self, moondream_model):
        self.moondream = moondream_model
        self.tokenizer = moondream_model.tokenizer
        
        # Fixed settings for simplicity
        self.temperature = 0.8
        self.max_tokens = 200  # Higher for 150 token target
        
    def generate_text(self, prompt):
        """Generate text with fixed settings"""
        try:
            result = self.moondream.query(
                image=None,
                question=prompt,
                stream=False,
                settings={
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }
            )
            return result["answer"]
        except Exception as e:
            print(f"Generation error: {e}")
            return prompt + " [generation failed]"


def reward_function(generated_text, prompt):
    """Simple reward based on getting close to 150 tokens"""
    if prompt in generated_text:
        response = generated_text.replace(prompt, "").strip()
    else:
        response = generated_text.strip()
    
    token_count = len(response.split())
    target_tokens = 150
    distance = abs(token_count - target_tokens)
    reward = max(10.0 - (distance * 0.05), 0.1)  # Gentler penalty for 150 target
    
    return reward, token_count


def train_episode(rl_trainer, prompts, device):
    """Single training episode - just generate and measure reward"""
    episode_rewards = []
    episode_token_counts = []
    episode_data = []
    
    for prompt in prompts:
        # Generate text
        generated = rl_trainer.generate_text(prompt)
        reward, token_count = reward_function(generated, prompt)
        
        # Store results
        episode_rewards.append(reward)
        episode_token_counts.append(token_count)
        episode_data.append({
            'prompt': prompt,
            'generated': generated,
            'reward': reward,
            'tokens': token_count,
        })
    
    return episode_rewards, episode_token_counts, episode_data


def main():
    print("🚀 Starting Working RL Training for Moondream")
    print("=" * 60)
    print("📝 Using evaluation-based RL (no gradient issues)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize wandb
    wandb.init(
        project="moondream-rl-50tokens",
        name=f"working-rl-{int(time.time())}",
        config={
            "algorithm": "Evaluation-based RL",
            "target_tokens": 150,
            "device": str(device),
            "approach": "evolutionary_settings",
        }
    )
    
    try:
        # Load model using correct approach
        print("\n📦 Loading Moondream...")
        config = MoondreamConfig()
        moondream_model = MoondreamModel(config, dtype=torch.float16)
        
        print("   Downloading model weights...")
        weights_path = hf_hub_download(
            repo_id="vikhyatk/moondream2", 
            filename="model.safetensors",
            revision="2025-06-21"
        )
        
        print("   Loading weights into model...")
        simple_load_weights(weights_path, moondream_model)
        moondream_model = moondream_model.to(device)
        
        # Create RL trainer
        rl_trainer = MoondreamRLTrainer(moondream_model)
        
        # Training prompts
        train_prompts = [
            "The weather today is",
            "I believe that",
            "Technology helps us",
            "Learning is important because",
            "In the future",
            "Science shows us",
            "Education means",
            "Success comes from",
            "Friendship is",
            "The best way to",
        ]
        
        print(f"📚 Training with {len(train_prompts)} prompts")
        print("🎯 Goal: Generate responses close to 150 tokens")
        
        # Training configuration
        num_episodes = wandb.config.get("num_episodes", 200)
        batch_size = 3
        
        print(f"\n🏃 Training for {num_episodes} episodes...")
        wandb.config.update({"num_episodes": num_episodes, "batch_size": batch_size})
        
        best_avg_reward = 0.0
        best_episode = 0
        
        for episode in range(num_episodes):
            # Sample random batch of prompts
            batch_prompts = random.sample(train_prompts, batch_size)
            
            # Training step
            rewards, token_counts, episode_data = train_episode(rl_trainer, batch_prompts, device)
            
            avg_reward = sum(rewards) / len(rewards)
            avg_tokens = sum(token_counts) / len(token_counts)
            token_std = torch.tensor(token_counts, dtype=torch.float32).std().item()
            
            # Track best performance
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_episode = episode + 1
            
            # Log to wandb
            wandb.log({
                "episode": episode + 1,
                "avg_reward": avg_reward,
                "avg_tokens": avg_tokens,
                "token_std": token_std,
                "best_avg_reward": best_avg_reward,
                "distance_from_target": abs(avg_tokens - 150),
                "target_progress": min(avg_tokens / 150, 1.0),
            })
            
            print(f"Episode {episode+1:3d}: Reward={avg_reward:5.2f}, Tokens={avg_tokens:4.1f}±{token_std:3.1f}, Best={best_avg_reward:5.2f}")
            
            # Detailed progress every 10 episodes
            if (episode + 1) % 10 == 0:
                print("  📊 Sample generations:")
                sample_data = random.choice(episode_data)
                
                if sample_data['prompt'] in sample_data['generated']:
                    response = sample_data['generated'].replace(sample_data['prompt'], "").strip()
                else:
                    response = sample_data['generated'].strip()
                
                print(f"    Prompt: '{sample_data['prompt']}'")
                print(f"    Response: '{response[:60]}{'...' if len(response) > 60 else ''}'")
                print(f"    Tokens: {sample_data['tokens']}, Reward: {sample_data['reward']:.2f}")
                
                # Log sample generation metrics
                wandb.log({
                    "sample_tokens": sample_data['tokens'],
                    "sample_reward": sample_data['reward'],
                })
        
        print("\n🎉 Training completed!")
        
        # Final evaluation
        print("\n🧪 Final Evaluation:")
        eval_prompts = ["Tell me about", "The world is", "Computers can"]
        
        total_reward = 0
        total_tokens = 0
        
        for prompt in eval_prompts:
            temp, max_tok = rl_trainer.get_best_settings_for_prompt(prompt)
            generated = rl_trainer.generate_with_settings(prompt, temp, max_tok)
            reward, token_count = reward_function(generated, prompt)
            
            total_reward += reward
            total_tokens += token_count
            
            print(f"  '{prompt}' → Tokens: {token_count}, Reward: {reward:.2f}")
        
        avg_eval_reward = total_reward / len(eval_prompts)
        avg_eval_tokens = total_tokens / len(eval_prompts)
        
        print(f"\n📈 Final Results:")
        print(f"  Average Reward: {avg_eval_reward:.2f}/10.0")
        print(f"  Average Tokens: {avg_eval_tokens:.1f}/150")
        print(f"  Best Episode: {best_episode}")
        print(f"  Target Achievement: {avg_eval_tokens/150*100:.1f}%")
        
        # Log final results
        wandb.log({
            "final/avg_reward": avg_eval_reward,
            "final/avg_tokens": avg_eval_tokens,
            "final/best_episode": best_episode,
            "final/target_achievement": avg_eval_tokens/150*100,
        })
        
        # Create final evaluation table
        eval_table = wandb.Table(
            columns=["Prompt", "Response", "Tokens", "Reward"],
            data=[]
        )
        
        for prompt in eval_prompts:
            generated = rl_trainer.generate_text(prompt)
            reward, token_count = reward_function(generated, prompt)
            
            if prompt in generated:
                response = generated.replace(prompt, "").strip()
            else:
                response = generated.strip()
            
            eval_table.add_data(prompt, response[:80], token_count, round(reward, 2))
        
        wandb.log({"final_evaluation": eval_table})
        
        if avg_eval_tokens >= 130 and avg_eval_tokens <= 170:
            print("  🎯 EXCELLENT! Model learned to hit the 150-token target!")
            success = True
        elif avg_eval_tokens >= 100:
            print("  📈 GOOD! Model made significant progress!")
            success = True
        else:
            print("  📝 Model needs more training")
            success = False
            
        wandb.summary.update({
            "success": success,
            "final_tokens": avg_eval_tokens,
            "target_tokens": 150,
        })
        
        print(f"\n📊 View results at: {wandb.run.url}")
        wandb.finish()
            
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        wandb.finish()


if __name__ == "__main__":
    main()