import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_dir = "distilgpt2-GRPO-finetuned"
    # Load the fine-tuned model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define a test prompt.
    test_prompt = (
        "Text: I cannot stop smiling because everything is wonderful!\n"
        "What emotion is expressed? Please think step by step and then provide your final answer in the format <think>...<answer>... ."
    )
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)

    # Generate a completion.
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=tokenizer.model_max_length,
        do_sample=True,
        temperature=0.9,
        num_return_sequences=1,
    )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Generated text:", generated_text)

if __name__ == "__main__":
    main()
