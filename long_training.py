"""
Long RL training run for Moondream
Extended training with more episodes and better hyperparameters
"""
import os
import sys

# Set wandb config for long run
os.environ["WANDB_PROJECT"] = "moondream-rl-50tokens"
os.environ["WANDB_RUN_NAME"] = "long-training-500-episodes"

# Import and run the main training
from rl_training import *

def long_training_main():
    print("🚀 Starting LONG RL Training for Moondream")
    print("=" * 70)
    print("🎯 Goal: 500 episodes to reach 50-token target")
    print("📊 Full Weights & Biases tracking enabled")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize wandb with long training config
    wandb.init(
        project="moondream-rl-50tokens",
        name=f"long-training-500ep-{int(time.time())}",
        config={
            "algorithm": "REINFORCE",
            "target_tokens": 50,
            "learning_rate_policy": 5e-5,  # Slightly lower for stability
            "learning_rate_value": 1e-3,
            "max_tokens": 60,
            "num_episodes": 500,  # Much longer run
            "batch_size": 4,  # Slightly larger batch
            "device": str(device),
            "notes": "Long training run to reach 50-token target",
        }
    )
    
    try:
        # Load model
        print("\n📦 Loading Moondream...")
        hf_config = HfConfig()
        moondream_config = MoondreamConfig()
        hf_config.config = moondream_config.to_dict()
        
        moondream_model = HfMoondream(hf_config)
        
        weights_path = hf_hub_download(
            repo_id="vikhyatk/moondream2", 
            filename="model.safetensors",
            revision="2025-06-21"
        )
        simple_load_weights(weights_path, moondream_model.model)
        
        # Create RL trainer
        rl_model = MoondreamRLTrainer(moondream_model)
        rl_model.to(device)
        
        # Optimizers with adjusted learning rates
        optimizer_policy = optim.Adam(rl_model.policy_net.parameters(), lr=5e-5)
        optimizer_value = optim.Adam(rl_model.value_net.parameters(), lr=1e-3)
        
        # Expanded training prompts for more diversity
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
            "People often think",
            "When I consider",
            "The most important thing",
            "Research indicates that",
            "Many experts believe",
            "Studies have shown",
            "It's interesting that",
            "One could argue that",
            "From my perspective",
            "Generally speaking",
        ]
        
        print(f"📚 Training with {len(train_prompts)} diverse prompts")
        print("🎯 Goal: Learn to generate exactly 50 tokens per response")
        
        # Training configuration
        num_episodes = 500
        batch_size = 4
        
        print(f"\n🏃 Starting LONG training run: {num_episodes} episodes...")
        print(f"📈 Expected training time: ~2-3 hours")
        
        best_reward = 0.0
        episodes_since_improvement = 0
        target_reached = False
        
        for episode in range(num_episodes):
            # Sample random batch of prompts
            batch_prompts = random.sample(train_prompts, batch_size)
            
            # Training step
            avg_reward, token_counts = train_rl_step(
                rl_model, optimizer_policy, optimizer_value, batch_prompts, device
            )
            
            avg_tokens = sum(token_counts) / len(token_counts)
            token_std = torch.tensor(token_counts, dtype=torch.float32).std().item()
            
            # Track best reward and improvement
            if avg_reward > best_reward:
                best_reward = avg_reward
                episodes_since_improvement = 0
            else:
                episodes_since_improvement += 1
            
            # Check if target reached (within 5 tokens of 50)
            if abs(avg_tokens - 50) <= 5 and not target_reached:
                target_reached = True
                print(f"🎉 TARGET NEARLY REACHED at episode {episode + 1}!")
                print(f"   Average tokens: {avg_tokens:.1f} (target: 50)")
            
            # Log to wandb
            wandb.log({
                "episode": episode + 1,
                "avg_reward": avg_reward,
                "avg_tokens": avg_tokens,
                "token_std": token_std,
                "best_reward": best_reward,
                "distance_from_target": abs(avg_tokens - 50),
                "target_progress": min(avg_tokens / 50, 1.0),
                "episodes_since_improvement": episodes_since_improvement,
                "target_reached": target_reached,
            })
            
            # Progress reporting
            if episode < 100 or (episode + 1) % 25 == 0:
                print(f"Episode {episode+1:3d}: Reward={avg_reward:5.2f}, Tokens={avg_tokens:4.1f}±{token_std:3.1f}, Best={best_reward:5.2f}")
            
            # Detailed progress every 50 episodes
            if (episode + 1) % 50 == 0:
                print(f"\n📊 Progress Report - Episode {episode + 1}")
                print(f"   Current Performance: {avg_tokens:.1f} tokens (target: 50)")
                print(f"   Best Reward So Far: {best_reward:.2f}")
                print(f"   Target Achievement: {min(avg_tokens/50*100, 100):.1f}%")
                
                # Sample generation
                test_prompt = random.choice(train_prompts)
                with torch.no_grad():
                    generated, _, _ = rl_model.generate_with_policy(test_prompt)
                    reward, token_count = reward_function(generated, test_prompt)
                    
                    if test_prompt in generated:
                        response = generated.replace(test_prompt, "").strip()
                    else:
                        response = generated.strip()
                    
                    print(f"   Sample: '{test_prompt}' → '{response[:50]}...'")
                    print(f"   Tokens: {token_count}, Reward: {reward:.2f}\n")
                    
                    # Log sample generation
                    wandb.log({
                        f"detailed_sample/episode_{episode+1}": {
                            "prompt": test_prompt,
                            "response": response[:100],
                            "tokens": token_count,
                            "reward": reward,
                        }
                    })
            
            # Early stopping if no improvement for 200 episodes
            if episodes_since_improvement > 200:
                print(f"\n⏰ Early stopping: No improvement for {episodes_since_improvement} episodes")
                break
        
        print("\n🎉 Long training completed!")
        
        # Comprehensive final evaluation
        print("\n🧪 Final Comprehensive Evaluation:")
        eval_prompts = [
            "Tell me about", "The world is", "Computers can", "Artificial intelligence",
            "Climate change", "Space exploration", "Medical research", "Future technology"
        ]
        
        total_reward = 0
        total_tokens = 0
        perfect_scores = 0
        
        for prompt in eval_prompts:
            with torch.no_grad():
                generated, _, _ = rl_model.generate_with_policy(prompt)
                reward, token_count = reward_function(generated, prompt)
                
                total_reward += reward
                total_tokens += token_count
                
                if abs(token_count - 50) <= 2:  # Within 2 tokens of target
                    perfect_scores += 1
                
                print(f"  '{prompt}' → Tokens: {token_count}, Reward: {reward:.2f}")
        
        avg_eval_reward = total_reward / len(eval_prompts)
        avg_eval_tokens = total_tokens / len(eval_prompts)
        accuracy = perfect_scores / len(eval_prompts) * 100
        
        print(f"\n📈 Final Results:")
        print(f"  Average Reward: {avg_eval_reward:.2f}/10.0")
        print(f"  Average Tokens: {avg_eval_tokens:.1f}/50") 
        print(f"  Target Accuracy: {accuracy:.1f}% (within ±2 tokens)")
        print(f"  Total Episodes: {episode + 1}")
        print(f"  Best Reward Achieved: {best_reward:.2f}")
        
        # Log comprehensive final results
        wandb.log({
            "final/avg_reward": avg_eval_reward,
            "final/avg_tokens": avg_eval_tokens,
            "final/target_accuracy": accuracy,
            "final/total_episodes": episode + 1,
            "final/best_reward_overall": best_reward,
            "final/target_achievement": avg_eval_tokens/50*100,
        })
        
        # Success criteria
        if accuracy >= 50:  # 50% of responses within ±2 tokens
            print("  🏆 EXCELLENT! Model successfully learned the target!")
            success_level = "excellent"
        elif avg_eval_tokens >= 40:  # At least 40 tokens on average  
            print("  🎯 GOOD! Model made significant progress!")
            success_level = "good"
        elif avg_eval_tokens >= 30:
            print("  📈 FAIR! Model learned to generate longer responses")
            success_level = "fair"
        else:
            print("  📝 Model needs more training or different approach")
            success_level = "needs_work"
            
        wandb.summary.update({
            "success_level": success_level,
            "final_tokens": avg_eval_tokens,
            "target_tokens": 50,
            "accuracy_within_2": accuracy,
            "total_episodes_run": episode + 1,
        })
        
        print(f"\n📊 View detailed results at: {wandb.run.url}")
        wandb.finish()
            
    except Exception as e:
        print(f"\n❌ Long training failed: {e}")
        import traceback
        traceback.print_exc()
        wandb.finish()


if __name__ == "__main__":
    long_training_main()