from typing import Dict, Any
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

# Define complexity levels and their corresponding k values
K_VALUES = {
    "simple_math": 4,        # Simple calculations only need a few attempts
    "easy_questions": 3,     # Basic questions with straightforward answers
    "general_knowledge": 7,  # Might need a few more tries for better accuracy
    "complex_reasoning": 10,  # More paths for complex problems
    "expert_topics": 10    # Slightly more paths for specialized topics
}

CLASSIFICATION_PROMPT = """Given a question, classify its complexity into one of these categories:
1. simple_math - Basic arithmetic and simple calculations
2. easy_questions - Basic stuff that is easy to answer
3. general_knowledge - Basic facts and common knowledge
4. complex_reasoning - Multi-step problems, logical reasoning, detailed explanations
5. expert_topics - Technical, scientific, or specialized topics

Question: {question}

Only replay either simple_math, general_knowledge, complex_reasoning, or expert_topics and nothing else.

Classification: """

def get_k_value(messages: list, context) -> tuple[int, str]:
    """
    Determine the appropriate k value based on question complexity using LLM classification.
    Returns: (k_value, classification_type)
    """
    model, tokenizer = context.on_start_value
    
    # Get the last user message
    user_message = next((msg["content"] for msg in reversed(messages) 
                        if msg["role"] == "user"), "")
    
    # Prepare the prompt
    prompt = CLASSIFICATION_PROMPT.format(question=user_message)
    messages = [{"role": "user", "content": prompt}]
    
    # Format with chat template
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Generate classification
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.1,  # Low temperature for more deterministic output
            num_return_sequences=1,
        )
        classification = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Extract the classification from the response
    classification = classification.strip().lower()
    print(classification)
    
    # Map to valid category and return both k value and classification type
    if "simple_math" in classification:
        return K_VALUES["simple_math"], "simple_math"
    elif "easy" in classification:
        return K_VALUES["easy_questions"], "easy_questions"
    elif "expert" in classification:
        return K_VALUES["expert_topics"], "expert_topics"
    elif "complex" in classification:
        return K_VALUES["complex_reasoning"], "complex_reasoning"
    else:
        return K_VALUES["general_knowledge"], "general_knowledge"