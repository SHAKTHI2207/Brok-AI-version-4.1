# Brok AI v4.1 — Final Version with Streamlit Interface
# Built by Shakthi | Powered by Mistral-7B Instruct | Persona-Driven Marketing Assistant

import os
import re
import torch
import random
import json
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import streamlit as st

# File paths
MEMORY_FILE = "brok_memory.json"
FINETUNE_FILE = "finetune_data.jsonl"
LOG_FILE = "brok_chat_log.txt"

# Initialize model & tokenizer
model = None
tokenizer = None
active_persona = "Default"

marketing_facts = [
    "92% of consumers trust user-generated content more than traditional advertising. (Nielsen)",
    "Emotional ads outperform rational ones by 31% in ROI. (IPA)",
    "Brands with consistent branding see revenue increase by 33%. (Lucidpress)",
    "Coca-Cola's 'Share a Coke' campaign increased sales by 2% after a decade of decline.",
    "Short-form videos have the highest ROI of any social media content. (HubSpot)"
]

def load_model():
    global model, tokenizer
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    max_memory = {0: 14e9, "cpu": 64e9}
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()

def content_mod(text):
    offensive = ["fuck off", "mother fucker", "bitch", "idiot", "stupid", "dumb"]
    for w in offensive:
        if re.search(rf"\\b{re.escape(w)}\\b", text, flags=re.IGNORECASE):
            return "Sir, I’m programmed for precision and decorum. Try a less colorful phrasing."
    return text

def stylize_response(text):
    tags = {
        "Ad Guru": "🧠 Brok (Ad Guru)",
        "Product Marketer": "📦 Brok (Product Marketer)",
        "Meme Generator": "😂 Brok (Meme Lord)",
        "Default": "🤖 Brok"
    }
    return f"{tags.get(active_persona, '🤖 Brok')}: {text}"

def generate_response(prompt):
    if not model or not tokenizer:
        return "⚠️ Model not loaded."
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.pad_token_id
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        return content_mod(text.split(prompt, 1)[-1].strip())
    except Exception as e:
        return f"😓 Brok hit an issue: {e}"

def handle_persona_switch(user_input):
    global active_persona
    persona_tags = {
        "[Ad Guru]": "Ad Guru",
        "[Product Marketer]": "Product Marketer",
        "[Meme Generator]": "Meme Generator",
        "[Default]": "Default"
    }
    for tag, persona in persona_tags.items():
        if tag in user_input:
            active_persona = persona
            return f"🎭 Persona switched to {persona}."
    return None

def get_prompt(user_input):
    facts = "\n".join(random.sample(marketing_facts, 2))
    return (
        f"You are Brok AI, a marketing strategist with persona {active_persona}. "
        f"Speak clearly, creatively, and persuasively. Avoid fluff.\n\n"
        f"{facts}\nUser: {user_input}\nAssistant:"
    )

def load_memory():
    if os.path.exists(MEMORY_FILE):
        return json.load(open(MEMORY_FILE, "r", encoding="utf-8"))
    return {"conversations": []}

def save_memory(convo):
    mem = load_memory()
    mem["conversations"].append({"timestamp": datetime.now().isoformat(), "conversation": convo})
    json.dump(mem, open(MEMORY_FILE, "w", encoding="utf-8"), indent=2)

def save_for_finetuning(prompt, response):
    with open(FINETUNE_FILE, "a", encoding="utf-8") as f:
        json.dump({"prompt": prompt.strip(), "completion": response.strip()}, f)
        f.write("\n")

def log_chat(prompt, response):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"User: {prompt}\nAssistant: {response}\n\n")
    save_memory([{"user": prompt, "assistant": response}])
    save_for_finetuning(prompt, response)

# 🚀 Streamlit UI
st.set_page_config(page_title="Brok AI v4.1", page_icon="🤖")
st.title("🤖 Brok AI v4.1 – Your Meme-Loving Marketing Assistant")
st.markdown("Built by **Shakthi** | Powered by Mistral-7B | Persona-Driven & Marketing-Packed")

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

load_model()

user_input = st.text_input("💬 Ask Brok something... (Add persona like [Ad Guru], [Meme Generator], etc.)")

if user_input:
    persona_note = handle_persona_switch(user_input)
    prompt = get_prompt(user_input)
    response = generate_response(prompt)
    styled = stylize_response(response)

    st.session_state.chat_log.append((user_input, styled))
    log_chat(user_input, styled)

# Show chat history
for u, a in reversed(st.session_state.chat_log):
    st.markdown(f"**You:** {u}")
    st.markdown(f"{a}")
