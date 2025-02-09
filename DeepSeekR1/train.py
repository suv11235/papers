import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from utils import preprocess_function, reward_function, GRPOModelWrapper

def main():
    # 1. Load a subset of the GoEmotions dataset.
    dataset = load_dataset("go_emotions", split="train[:100]")
    label_names = dataset.features["labels"].feature.names

    # 2. Preprocess the dataset to create prompts and ground truth labels.
    processed_dataset = dataset.map(lambda x: preprocess_function(x, label_names))
    print("Sample after preprocessing:", processed_dataset[0])

    # 3. Load the tokenizer and model.
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    original_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = GRPOModelWrapper(original_model)

    # 4. Configure GRPO training parameters.
    training_args = GRPOConfig(
        output_dir="distilgpt2-GRPO",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_generations=4,
        max_prompt_length=128,
        max_completion_length=64,
        learning_rate=5e-6,
        num_train_epochs=1,
        logging_steps=10,
        use_vllm=False,
        save_safetensors=False,  # Disable safetensors for this demo.
    )

    # 5. Initialize the GRPO trainer.
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,
        args=training_args,
        train_dataset=processed_dataset,
        processing_class=tokenizer,
    )

    # 6. Start training.
    trainer.train()

    # 7. Save the fine-tuned model and tokenizer.
    trainer.model.save_pretrained("distilgpt2-GRPO-finetuned")
    tokenizer.save_pretrained("distilgpt2-GRPO-finetuned")
    print("Training complete. Model saved to 'distilgpt2-GRPO-finetuned'.")

if __name__ == "__main__":
    main()
