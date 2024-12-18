import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Tuple, Dict, Optional
import numpy as np

# script by https://github.com/codelion/optillm/blob/main/optillm/cot_decoding.py
# an implementation from the paper https://arxiv.org/abs/2402.10200 

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def calculate_confidence(logits: List[torch.Tensor], answer_ids: torch.Tensor) -> float:
    """
    Calculate the confidence score (Δ) as specified in the paper.
    
    Args:
        logits: List of logits for each decoding step
        answer_ids: Tensor of token ids for the answer
    
    Returns:
        Confidence score (Δ)
    """
    confidence_sum = 0.0
    valid_tokens = 0
    for t, token_id in enumerate(answer_ids):
        if t >= len(logits):
            break
        token_logits = logits[t]
        probs = torch.softmax(token_logits, dim=-1)
        if probs.size(-1) > 1:
            top_2_probs, _ = torch.topk(probs, min(2, probs.size(-1)))
            if top_2_probs.size(-1) > 1:
                confidence_sum += (top_2_probs[-1][0] - top_2_probs[-1][1]).item()
            else:
                confidence_sum += 1.0  # Max confidence if there's only one token
        else:
            confidence_sum += 1.0  # Max confidence if there's only one token
        valid_tokens += 1
    
    return confidence_sum / valid_tokens if valid_tokens > 0 else 0.0

def aggregate_paths_based_on_scores(paths: List[Tuple[str, float]]) -> Tuple[str, float]:
    """Aggregate multiple paths based on their confidence scores."""
    answer_scores = {}
    for answer, delta in paths:
        answer_scores[answer] = answer_scores.get(answer, 0) + delta
    best_answer = max(answer_scores, key=answer_scores.get)
    return best_answer, answer_scores[best_answer]

def cot_decode(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    k: int = 10,
    num_beams: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    early_stopping: bool = False,
    aggregate_paths: bool = False,
) -> Tuple[str, float, int]:
    """
    Implement CoT-decoding for a given chat input.
    Returns: (output_text, confidence_score, llm_calls)
    """
    device = get_device()
    model.to(device)
    
    llm_calls = 0  # Initialize counter

    # First LLM call for initial logits
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        first_token_logits = outputs.logits[0, -1, :]
        top_k_logits, top_k_indices = torch.topk(first_token_logits, k)

    paths = []
    for idx in top_k_indices:
        # Generate sequence starting with the selected token
        start_ids = torch.cat([input_ids, idx.unsqueeze(0).unsqueeze(0)], dim=-1)
        start_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=-1)
        
        output = model.generate(
            start_ids,
            attention_mask=start_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        llm_calls += 1  # One call per path
        
        generated_sequence = output.sequences[0]
        answer_ids = generated_sequence[len(input_ids[0]):]
        answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
        
        confidence = calculate_confidence(output.scores, answer_ids)
        paths.append((answer_text, confidence))
    
    if aggregate_paths:
        best_answer, confidence = aggregate_paths_based_on_scores(paths)
    else:
        best_answer, confidence = max(paths, key=lambda x: x[1])
        
    return best_answer, confidence, llm_calls