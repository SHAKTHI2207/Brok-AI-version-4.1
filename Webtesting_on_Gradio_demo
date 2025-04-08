############ BROK AI FIXED VERSION for Deployment ########
# Brok AI v4.1 — Jarvis x TARS x MemeLord Fusion with Marketing Power
# Built by Shakthi | Runs on Mistral-7B Instruct | Persona-Driven Assistant.
##### Built on Gradio UI interface

import os
import re
import torch
import random
import json
from datetime import datetime
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# === File Paths ===
MEMORY_FILE = "brok_memory.json"
FINETUNE_FILE = "finetune_data.jsonl"
LOG_FILE = "brok_chat_log.txt"

# === Global Variables ===
marketing_facts = [
    "92% of consumers trust user-generated content more than traditional advertising. (Nielsen)",
    "Emotional ads outperform rational ones by 31% in ROI. (IPA)",
    "Brands with consistent branding see revenue increase by 33%. (Lucidpress)",
    "Coca-Cola's 'Share a Coke' campaign increased sales by 2% after a decade of decline.",
    "Short-form videos have the highest ROI of any social media content. (HubSpot)"
]

# === Load Model ===# 🚀 Load Mistral-7B with 4-bit Quantization and Memory Management
def load_model_and_tokenizer():
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
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# === Prompt & Response Generation ===

def get_prompt(user_input, persona):
    facts = "\n".join(random.sample(marketing_facts, 2))
    return (
        f"You are Brok AI, a marketing strategist with the persona: {persona}. "
        f"Communicate with clarity, creativity, and punch. No fluff.\n\n"
        f"{facts}\nUser: {user_input}\nAssistant:"
    )
# 🚫 Content Moderation Filter

def content_mod(text):
    offensive = ["fuck off", "mother fucker", "bitch", "idiot", "stupid", "dumb"]
    for word in offensive:
        if re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE):
            return "🚫 Let’s keep it respectful. Brok believes in pro convos only."
    return text

# 🎭 Persona Management
def stylize_response(text, persona):
    tags = {
        "Ad Guru": "🧠 Brok (Ad Guru)",
        "Product Marketer": "📦 Brok (Product Marketer)",
        "Meme Generator": "😂 Brok (Meme Lord)",
        "Default": "🤖 Brok"
    }
    return f"{tags.get(persona, '🤖 Brok')}: {text}"

  # 🧠 Generate Model Response

def generate_response(message, chat_history, persona):
    prompt = get_prompt(message, persona)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.pad_token_id
    )
    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    result = content_mod(raw.split("Assistant:")[-1].strip())
    styled = stylize_response(result, persona)
    log_chat(message, styled)
    return styled

# 🗂 Memory Handling

def save_memory(convo):
    mem = json.load(open(MEMORY_FILE, "r", encoding="utf-8")) if os.path.exists(MEMORY_FILE) else {"conversations": []}
    mem["conversations"].append({"timestamp": datetime.now().isoformat(), "conversation": convo})
    json.dump(mem, open(MEMORY_FILE, "w", encoding="utf-8"), indent=2)

 #📝 Collect Data for Fine-Tuning

def save_for_finetuning(prompt, response):
    with open(FINETUNE_FILE, "a", encoding="utf-8") as f:
        json.dump({"prompt": prompt.strip(), "completion": response.strip()}, f)
        f.write("\n")
# 📑 Log Chat & Memory
def log_chat(prompt, response):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"User: {prompt}\nAssistant: {response}\n\n")
    save_memory([{"user": prompt, "assistant": response}])
    save_for_finetuning(prompt, response)

# === Gradio UI ===

with gr.Blocks(css="""
body {
    background: url('/mnt/data/A_2D_digital_interface_design_showcases_"Brok_AI,".png') no-repeat center center fixed;
    background-size: cover;
    font-family: 'Segoe UI', sans-serif;
    color: white;
}

#brok-header {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #00FFE0;
    text-shadow: 0px 0px 15px rgba(0, 255, 224, 0.8);
    margin-top: 20px;
    margin-bottom: 30px;
}

.persona-btn {
    margin: 0 10px;
    padding: 10px 20px;
    border: none;
    background-color: rgba(0, 0, 0, 0.4);
    color: #00FFE0;
    border-radius: 12px;
    transition: all 0.3s ease;
    font-weight: bold;
    backdrop-filter: blur(10px);
}

.persona-btn:hover {
    background-color: rgba(0, 255, 224, 0.15);
    color: white;
    cursor: pointer;
    transform: scale(1.05);
    box-shadow: 0 0 10px #00FFE0;
}

.markdown-text {
    font-size: 20px;
    text-align: center;
    margin-bottom: 10px;
}

.chatbox, .message {
    background-color: rgba(0, 0, 0, 0.6) !important;
    border-radius: 15px !important;
    color: #E0F7FA !important;
    padding: 12px !important;
}

""") as demo:

    gr.HTML("<div id='brok-header'>🤖 <b>Brok AI</b> — The Marketing Intelligence Chatbot</div>")
    
    persona_state = gr.State("Default")

    with gr.Row():
        gr.Markdown("**🎭 Choose a Persona:**", elem_classes="markdown-text")
    
    with gr.Row():
        def set_persona(p): return p
        gr.Button("🧠 Ad Guru", elem_classes="persona-btn").click(lambda: "Ad Guru", None, persona_state)
        gr.Button("📦 Product Marketer", elem_classes="persona-btn").click(lambda: "Product Marketer", None, persona_state)
        gr.Button("😂 Meme Generator", elem_classes="persona-btn").click(lambda: "Meme Generator", None, persona_state)
        gr.Button("🤖 Default", elem_classes="persona-btn").click(lambda: "Default", None, persona_state)

    def chat_fn(message, history, persona):
        response = generate_response(message, history, persona)
        return response

    gr.ChatInterface(
        fn=chat_fn,
        additional_inputs=[persona_state],
        chatbot=gr.Chatbot(label="💬 Chat with Brok AI"),
        title="Start chatting with Brok!",
        theme="soft",
    )

demo.launch()

