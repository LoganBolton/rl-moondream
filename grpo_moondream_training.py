import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from PIL import Image
import wandb
import os

# Initialize wandb
wandb.init(project="grpo-moondream-demo")

dataset = load_dataset("mlabonne/smoltldr")

# Load Moondream model
model_id = "vikhyatk/moondream2"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision="2025-06-21",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Setup LoRA configuration
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules="all-linear",  # Use all linear layers for simplicity
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())

# Reward function to encourage 50-token responses
ideal_length = 50

def reward_len(completions, **kwargs):
    """
    Reward function that penalizes responses that deviate from the ideal length
    """
    rewards = []
    
    for completion in completions:
        # Tokenize the completion to get actual token count
        tokens = tokenizer.encode(completion, add_special_tokens=False)
        token_count = len(tokens)
        
        # Calculate reward based on distance from ideal length
        # Use exponential decay for more aggressive length control
        length_penalty = abs(ideal_length - token_count)
        reward = -length_penalty  # Negative penalty becomes reward
        
        rewards.append(reward)
    
    return rewards



# Training arguments
training_args = GRPOConfig(
    output_dir="./grpo_moondream_50_tokens",
    learning_rate=1e-5,  # Lower learning rate for vision-language model
    per_device_train_batch_size=8,  # Smaller batch size for multimodal model
    gradient_accumulation_steps=4,  # Increase to maintain effective batch size
    max_prompt_length=512,
    max_completion_length=96,
    num_generations=4,  # Fewer generations due to computational cost
    optim="adamw_8bit",
    num_train_epochs=1,
    bf16=True,
    report_to=["wandb"],
    remove_unused_columns=False,
    logging_steps=10,
    save_steps=500,
    eval_steps=100,
    warmup_steps=50,
    gradient_checkpointing=False,  # Moondream doesn't support gradient checkpointing
    dataloader_num_workers=0,  # Avoid multiprocessing issues
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    reward_funcs=[reward_len],
)

# Print training info
print("Starting GRPO training for Moondream...")
print(f"Target response length: {ideal_length} tokens")
print(f"Training on {len(dataset['train'])} examples")

trainer.train()

# Save the final model
trainer.save_model("./grpo_moondream_50_tokens_final")

print("Training completed!")
print("Model saved to ./grpo_moondream_50_tokens_final")
