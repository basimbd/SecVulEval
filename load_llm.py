import os
import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from openai import OpenAI
from anthropic import Anthropic

client_oa = OpenAI()
client_anthr = Anthropic()

MODEL_MAX_LENGTH = 6144

def get_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        model_max_length=MODEL_MAX_LENGTH,
        truncation=True,
        trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side="left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def get_model(model_id: str):
    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     # bnb_4bit_quant_type="nf4",
    #     # bnb_4bit_compute_dtype=torch.bfloat16,
    #     # bnb_4bit_use_double_quant=True,
    # )
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        # quantization_config=bnb_config,
        # torch_dtype=torch.float16,
        torch_dtype=torch.bfloat16,
        max_position_embeddings=MODEL_MAX_LENGTH,
        device_map="auto",
        # low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir='/local/home/mbahmed/install_dir/.cache/huggingface/hub')#.to('cuda:0')

def get_pipe(model, tokenizer):
    torch.cuda.empty_cache()
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        truncation=True,
        max_new_tokens=1024,
        temperature=0.1,
        do_sample=True,
        device_map="auto",
        use_cache=True,
    )

def get_remote_llm_result(model_id: str, messages: list, temperature: float = 0.1, save_path: str = "remote_llm_raw.jsonl"):
    if model_id.startswith("gpt"):
        response = client_oa.chat.completions.create(
            model=model_id,
            max_tokens=1024,
            messages=messages,
            temperature=temperature
        )
        output = response.choices[0].message.content.strip()
        os.makedirs(f"{model_id.split("/")[-1]}/raw", exist_ok=True)
        with open(f"{model_id.split("/")[-1]}/raw/{save_path}", "a") as f:
            json.dump({"raw": output}, f, ensure_ascii=False)
            f.write("\n")
        return output
    elif model_id.startswith("claude"):
        toks = json.loads(client_anthr.messages.count_tokens(
            model=model_id,
            system=messages[0]["content"],
            messages=messages[1:]
        ).model_dump_json())["input_tokens"]
        # if toks > 20000:
        #     raise ValueError(f"Input too long for model {save_path}: {toks} tokens")
        message = client_anthr.messages.create(
            model=model_id,
            max_tokens=1024,
            system=messages[0]["content"],
            messages=messages[1:],
            temperature=temperature
        )
        if toks >=18000:
            time.sleep(50)
        output = message.content[0].text.strip()
        os.makedirs(f"{model_id.split("/")[-1]}/raw", exist_ok=True)
        with open(f"{model_id.split("/")[-1]}/raw/{save_path}", "a") as f:
            json.dump({"raw": output}, f, ensure_ascii=False)
            f.write("\n")
        return output
    else:
        raise ValueError(f"Unsupported remote model_id: {model_id}")

