################# ------------------------------------------------------------------------------
# BROK AI Project - Part of Brokverse™ Architecture
# Copyright (c) 2025 Shakthi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------
# Brok AI v4.1 — Jarvis x TARS x MemeLord Fusion with Marketing Power
# Built by Shakthi | Runs on Mistral-7B Instruct | Persona-Driven Assistant.

import os
import re
import torch
import random
import json
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 📂 Memory & Fine-Tune Data Files
MEMORY_FILE = "brok_memory.json"
FINETUNE_FILE = "finetune_data.jsonl"
LOG_FILE = "brok_chat_log.txt"

# 🔧 Global Variables
model = None
tokenizer = None
active_persona = "Default"

# 💡 Marketing Insights Pool
marketing_facts = [
    "92% of consumers trust user-generated content more than traditional advertising. (Nielsen)",
    "Emotional ads outperform rational ones by 31% in ROI. (IPA)",
    "Brands with consistent branding see revenue increase by 33%. (Lucidpress)",
    "Coca-Cola's 'Share a Coke' campaign increased sales by 2% after a decade of decline.",
    "Short-form videos have the highest ROI of any social media content. (HubSpot)"
]

# 🚀 Load Mistral-7B with 4-bit Quantization and Memory Management
def load_model():
    global model, tokenizer
    try:
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

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

# 🚫 Content Moderation Filter
def content_mod(text: str) -> str:
    offensive = ["fuck off", "mother fucker", "bitch", "idiot", "stupid", "dumb"]
    for w in offensive:
        if re.search(rf"\b{re.escape(w)}\b", text, flags=re.IGNORECASE):
            return "Sir, I’m programmed for precision and decorum. Try a less colorful phrasing."
    return text

# 🎭 Persona Management
def apply_persona(user_input: str) -> str:
    global active_persona
    mapping = {
        "[Ad Guru]": "Ad Guru",
        "[Product Marketer]": "Product Marketer",
        "[Meme Generator]": "Meme Generator",
        "[Default]": "Default"
    }
    for key, persona in mapping.items():
        if key in user_input:
            active_persona = persona
            messages = {
                "Ad Guru": "🧠💼 Ad Guru persona activated.",
                "Product Marketer": "📦📈 Product Marketer mode on.",
                "Meme Generator": "😂🔥 Meme Generator here.",
                "Default": "🤖 Default Brok back in control."
            }
            return messages[persona]
    return None

# ✨ Stylize Response with Persona Tag
def stylize_response(text: str) -> str:
    tags = {
        "Ad Guru": "🧠 Brok (Ad Guru)",
        "Product Marketer": "📦 Brok (Product Marketer)",
        "Meme Generator": "😂 Brok (Meme Lord)",
        "Default": "🤖 Brok"
    }
    return f"{tags.get(active_persona, '🤖 Brok')}: {text}"

# 🧠 Generate Model Response
def generate_response(prompt: str) -> str:
    if not model or not tokenizer:
        return "⚠️ Model not loaded properly."

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

# 🗂 Memory Handling
def load_memory():
    if os.path.exists(MEMORY_FILE):
        return json.load(open(MEMORY_FILE, "r", encoding="utf-8"))
    return {"conversations": []}

def save_memory(conversation):
    mem = load_memory()
    mem["conversations"].append({
        "timestamp": datetime.now().isoformat(),
        "conversation": conversation
    })
    json.dump(mem, open(MEMORY_FILE, "w", encoding="utf-8"), indent=2)

# 📝 Collect Data for Fine-Tuning
def save_for_finetuning(user_input: str, response: str):
    with open(FINETUNE_FILE, "a", encoding="utf-8") as f:
        rec = {"prompt": user_input.strip(), "completion": response.strip()}
        f.write(json.dumps(rec) + "\n")

# 📑 Log Chat & Memory
def log_chat(user_input: str, response: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"User: {user_input}\nAssistant: {response}\n\n")
    save_memory([{"user": user_input, "assistant": response}])
    save_for_finetuning(user_input, response)

# 🎮 Command Handler
def handle_command(cmd: str) -> str:
    global history, active_persona
    if cmd.startswith("/reset"):
        history = []
        return "🧹 Brok: Chat history cleared."
    elif cmd.startswith("/persona"):
        _, *rest = cmd.split()
        name = " ".join(rest).title()
        if name in ["Ad Guru", "Product Marketer", "Meme Generator", "Default"]:
            active_persona = name
            return f"🎭 Brok switched to {active_persona} mode."
        else:
            return "❓ Brok: Unknown persona. Available: Ad Guru, Product Marketer, Meme Generator, Default."
    elif cmd.startswith("/facts"):
        return "📊 Random Marketing Facts:\n- " + "\n- ".join(random.sample(marketing_facts, 2))
    return None

# 🎬 Main Loop
if __name__ == "__main__":
    model, tokenizer = load_model()
    print("🤖 Brok AI: Online and ready with memory & learning!")

    history = []
    while True:
        user = input("You: ")
        if user.lower() in ["exit", "bye"]:
            print("🤖 Brok AI: Goodbye! Keep innovating.")
            break

        # Commands
        if user.startswith("/"):
            cmd_output = handle_command(user)
            if cmd_output:
                print(cmd_output)
                continue

        # Persona switch
        pf = apply_persona(user)
        if pf:
            print(pf)
            continue

        # Append to history
        history.append({"user": user})
        if len(history) > 10:
            history = history[-10:]

        # Build recent memory context
        mem = load_memory().get("conversations", [])[-3:]
        recent = "".join(
            f"User: {c['conversation'][0]['user']}\nAssistant: {c['conversation'][0]['assistant']}\n"
            for c in mem
        )

        # Prepare prompt
        facts = "\n".join(random.sample(marketing_facts, 2))
        convo = "".join(f"User: {h['user']}\n" for h in history)
        prompt = (
            f"You are Brok AI, a marketing strategist with persona {active_persona}. "
            "Speak clearly, creatively, and persuasively. Avoid fluff.\n\n"
            f"{recent}\n{facts}\n{convo}Assistant:"
        )

        # Generate & respond
        resp = generate_response(prompt)
        styled = stylize_response(resp)
        print(styled)
        log_chat(user, styled)
