import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from google.colab import userdata
from peft import PeftModel
from datasets import load_dataset
import pandas as pd


peft_model_path ="./gemini-fine-tuned"
device = "cuda:0"
model_id = "google/gemma-2b"
max_length = 1024
tokenizer = AutoTokenizer.from_pretrained(model_id, token=userdata.get('HF_TOKEN'))

cot_prompt_eval = """You will be provided with a single user input containing both context information and a question. Use the context to generate a single, cohesive rationale that fully answers the question.

    **User Input:**: {userInput}

    ### Instructions:
    1. Carefully review the context information to identify the most relevant details that directly address the question.
    2. Synthesize the information from the context with general knowledge to construct a single, unified rationale.
    3. Explain your reasoning clearly, integrating key points from the context to support your answer.
    4. Ensure that the rationale is thorough, logical, and directly answers the question in a well-supported way.

    **Rationale:**:
    """

def get_base_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, token=userdata.get('HF_TOKEN'))
    return model

def get_fine_tuned_modelr(model):
    lora_model = PeftModel.from_pretrained(model, peft_model_path)
    eval_tokenizer = AutoTokenizer.from_pretrained(peft_model_path, trust_remote_code=True)
    return lora_model,eval_tokenizer

def formatting_func(example):
    text = cot_prompt_eval.format(userInput=example["source"],rationale=example["rationale"],answer=example["target"])
    return text

def generate_and_tokenize_prompt(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


def get_dataset_for_eval():
    ds = load_dataset("kaist-ai/CoT-Collection")
    test_dataset = ds["train"].shuffle(seed=2).select([i for i in range(50)])
    tokenized_test_dataset = test_dataset.map(generate_and_tokenize_prompt).remove_columns(['source', 'target', 'rationale', 'task', 'type'])
    return tokenized_test_dataset

def perform_inference():
    df = pd.DataFrame(columns=['ModelOutputFine-tuned', 'BaseModel'])
    tokenized_test_dataset=get_dataset_for_eval()
    #base model inference
    model = get_base_model()
    base_model_out_tok = model.generate(torch.tensor(tokenized_test_dataset["input_ids"]).cuda(), max_new_tokens=64)
    base_model_out=tokenizer.batch_decode(torch.tensor(base_model_out_tok), skip_special_tokens=True)
    df["BaseModel"]=base_model_out
    #fine-tune-model-inference
    lora_model=get_fine_tuned_modelr()
    fine_tune_model_out_tok = lora_model.generate(torch.tensor(tokenized_test_dataset["input_ids"]).cuda(), max_new_tokens=64)
    fine_tune_model_out = tokenizer.batch_decode(torch.tensor(fine_tune_model_out_tok), skip_special_tokens=True)
    df["ModelOutputFine-tuned"]=fine_tune_model_out
    df.to_excel('./final.xlsx', index=False)


if __name__=="__main__":
    perform_inference()





    








