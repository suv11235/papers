import re
import torch.nn as nn

class GRPOModelWrapper(nn.Module):
    """
    Wraps a causal language model so that it is compatible with the GRPO trainer.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Expose the model's configuration.
        self.config = model.config

    def forward(self, *args, **kwargs):
        # Remove unsupported keyword if present.
        kwargs.pop("num_logits_to_keep", None)
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        # Delegate attribute lookup to the wrapped model.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

def preprocess_function(example, label_names):
    """
    Preprocess a single example from the GoEmotions dataset.

    Constructs a prompt that asks the model to think step-by-step and provide
    an answer in the format <think>...</think><answer>...</answer>.

    Returns:
        A dictionary with:
          - "prompt": the formatted prompt.
          - "ground_truth": the emotion label (using the first label if available,
            otherwise 'neutral').
    """
    text = example["text"]
    prompt = (
        f"Text: {text}\n"
        "What emotion is expressed? Please think step by step and then provide your final answer "
        "in the format <think>...<answer>... ."
    )
    if len(example["labels"]) > 0:
        gt_label = label_names[example["labels"][0]]
    else:
        gt_label = "neutral"
    return {"prompt": prompt, "ground_truth": gt_label}

def reward_function(completions, ground_truth, **kwargs):
    """
    Custom reward function for GRPO training.

    Each generated completion is expected to follow the format:
      <think> ... </think><answer> ... </answer>
    
    The function extracts the text within <answer> tags and compares it (ignoring case)
    to the ground truth emotion. It returns 1.0 if they match, 0.0 otherwise.
    """
    rewards = []
    pattern = r"<think>.*?</think><answer>(.*?)</answer>"
    for comp, gt in zip(completions, ground_truth):
        match = re.search(pattern, comp, re.DOTALL)
        if match:
            answer = match.group(1).strip().lower()
            rewards.append(1.0 if answer == gt.lower() else 0.0)
        else:
            rewards.append(0.0)
    return rewards
