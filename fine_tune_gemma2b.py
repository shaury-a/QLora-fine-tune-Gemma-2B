import os
import torch
import logging
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Logging Configuration
logging.basicConfig(filename='gemma2b_finetune.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_ID = "google/gemma-2b"
MAX_LENGTH = 768

# Load Tokenizer and Model
logger.info("Initializing tokenizer and model.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.getenv('HF_API_KEY'))
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map={"": 0}, token=os.getenv('HF_API_KEY')
)
logger.info("Tokenizer and model initialized successfully.")


# Prompt Template
def get_cot_prompt():
    """
    Defines the prompt template for Chain-of-Thought (CoT) fine-tuning.
    """
    logger.info("Generating Chain-of-Thought prompt.")
    return """You will be provided with a single user input containing both context information and a question. Use the context to generate a single, cohesive rationale that fully answers the question.

    **User Input:**: {userInput}

    ### Instructions:
    1. Carefully review the context information to identify the most relevant details that directly address the question.
    2. Synthesize the information from the context with general knowledge to construct a single, unified rationale.
    3. Explain your reasoning clearly, integrating key points from the context to support your answer.
    4. Ensure that the rationale is thorough, logical, and directly answers the question in a well-supported way.

    **Rationale:**: {rationale}

    **Answer:**: {answer}
    """


# Dataset Processing
def formatting_func(example):
    """
    Formats the dataset example into a CoT prompt.
    """
    return get_cot_prompt().format(
        userInput=example["source"],
        rationale=example["rationale"],
        answer=example["target"],
    )


def generate_and_tokenize_prompt(prompt):
    """
    Tokenizes the formatted prompt for training.
    """
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


def get_dataset(num_train_examples=1500, num_eval_examples=50):
    """
    Loads and processes the dataset for fine-tuning.
    """
    logger.info("Loading and processing the dataset.")
    ds = load_dataset("kaist-ai/CoT-Collection")
    train_dataset = ds["train"].shuffle(seed=42).select(range(num_train_examples))
    eval_dataset = ds["train"].shuffle(seed=42).select(range(num_eval_examples))
    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt).remove_columns(
        ['source', 'target', 'rationale', 'task', 'type']
    )
    tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt).remove_columns(
        ['source', 'target', 'rationale', 'task', 'type']
    )
    return tokenized_train_dataset, tokenized_val_dataset


# Logging Model Parameters
def log_trainable_parameters(model):
    """
    Logs the number of trainable parameters in the model.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable_params} || All params: {all_params} || Trainable%: {100 * trainable_params / all_params:.2f}%"
    )


# Fine-Tuning Function
def fine_tune_model():
    """
    Fine-tunes the gemma-2b model using LoRA (Low-Rank Adaptation).
    """
    load_dotenv()

    # Prepare Model for LoRA
    logger.info("Preparing model for fine-tuning.")
    model.gradient_checkpointing_enable()
    peft_model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=4,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(peft_model, lora_config)
    log_trainable_parameters(peft_model)

    # Load Dataset
    train_dataset, eval_dataset = get_dataset()

    # Training Configuration
    trainer = Trainer(
        model=peft_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=TrainingArguments(
            output_dir="./fine_tuned_model",
            warmup_steps=16,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            max_steps=100,
            learning_rate=2.5e-5,
            optim="paged_adamw_8bit",
            logging_steps=5,
            save_steps=25,
            eval_steps=25,
            save_strategy="steps",
            eval_strategy="steps",
            do_eval=True,
            report_to=None,
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Training
    logger.info("Starting training.")
    trainer.train()
    logger.info("Training completed.")

    # Save Fine-Tuned Model
    peft_model_path = "./fine_tuned_gemma"
    trainer.model.save_pretrained(peft_model_path)
    tokenizer.save_pretrained(peft_model_path)
    logger.info("Model and tokenizer saved successfully.")


if __name__ == "__main__":
    fine_tune_model()
