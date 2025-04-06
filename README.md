# Brok-AI-version-4.1
Say hello to Brok AI v4.1 – My Personal Marketing Assistant 🤖
---
# 🤖 Brok AI v4.1 – The Marketing Intelligence Chatbot

Welcome to **Brok AI**, a powerful, AI-powered marketing assistant built for intelligent content generation, strategic analysis, and creative brand engagement.

---

##  Project Summary
**Brok AI** is an experimental AI chatbot that evolved from GPT-2 into an advanced Mistral 7B-based model, trained to assist with marketing campaigns, content strategy, and smart communication. It’s your creative partner for marketing innovation.

---

##  The Journey: From GPT-2 to Mistral 7B

Brok didn’t start off this smart. Here’s the evolution:

### 1.GPT-2 Small
- First version of Brok was built using **GPT-2 small**.
- Faced **technical limitations** like low coherence, weak generation capabilities.
- Not suitable for generating marketing-grade content.

### 2.GPT-2 Large
- Tried scaling up with GPT-2 Large.
- Faced issues with:
  - **Tokenization errors**
  - **Temperature & sampling tuning**
  - **Content moderation mismatches**
  - **Inconsistent output**

### 3.GPT-J 6B
- Switched to GPT-J for better reasoning.
- Results improved in logic, but failed in:
  - **Maintaining marketing tone**
  - **Generating relevant creative content**
  - Gave **heavy but improper responses**

###  Mistral 7B (Current Model)
- Finally migrated to **Mistral 7B**, the current backbone of Brok.
- Delivers:
  - **Consistent marketing-grade output**
  - **Better response control**
  - **Realistic and usable generation**

➡ Despite compute limitations, this version is the most stable and production-ready.

---

💻 System Requirements

To run Brok AI v4.1 smoothly:
Component	Requirement
GPU	NVIDIA T4 or higher (A100, 3090, etc.)
VRAM	14–16 GB minimum
Precision	4-bit quantization + float16
CPU	Fallback possible, but slow
RAM	At least 16 GB recommended

---

## 📁 Files
- `app.py` → Main code 
- `README.md` → Overview & journey
- `requirements.txt` → Required dependencies
- `making_brok_ai.txt` → Dev struggles + model transition doc
- '

---

## 🔧 Installation
```bash
pip install -r requirements.txt
python app.py
```

---

## 🧪 Features
- Smart marketing content generation
- Conversation-ready chatbot
- High-quality response tuning
- Customizable prompts

---

## ⚠️ Known Limitations
- **Zero GPU = slower performance**
- Currently optimized for **CPU** only (bitsandbytes error on CUDA)
- Gated model access might require manual authentication

---

## 💡 Coming Soon
- Full conversational memory
- API integration with campaign tools
- Persona-switching for brands

---

## 📣 Creator
**Shakthi** – Marketing strategist + AI enthusiast

📍Built for dreamers. Optimized for marketers. Inspired by hustle.

> “Brok was born from trial, error, frustration, and passion. From GPT-2 to Mistral 7B — every line of code has a backstory.”

---

## 🤝 License
MIT License – Use, fork, improve.

 Coming Soon v5.0 

 Full chat memory
Brand persona switching
API + Campaign integration
Marketing analytics dashboard

---

### 📌 Note:

This README is just a glimpse. Check `struggles_brok_ai.txt` for the full development story, technical breakdown, and lessons learned.
Brok Ai is only accesible with T4 or greater GPU.

Brok 4.1 replaced version 3.5 due to its significant upgrades in capability, intelligence, and contextual understanding, made possible by the integration of the Mistral-7B Instruct model.

However, because this model contains over 7 billion parameters and operates using advanced 4-bit quantization with float16 precision, 
it demands a GPU with high VRAM and efficient tensor core support. GPUs like the NVIDIA T4 and higher (e.g., A100, V100, 3090) are required to run Brok 4.1 smoothly, 
as they offer the necessary memory bandwidth, VRAM (at least 14–16GB), and hardware acceleration to handle real-time inference. Lower-end GPUs, such as the 3050 or older Tesla cards, 
typically lack the resources to run such models reliably, leading to instability or failure during execution.
Thus, Brok 4.1 is optimized for T4 or higher to maintain high performance, faster token generation, and an overall responsive assistant experience.
