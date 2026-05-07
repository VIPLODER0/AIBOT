# 🤖 Advanced AI Telegram Bot

Multi-AI provider Telegram bot — OpenAI, Claude, Gemini, Groq, Mistral, Together, Cohere support ke saath.

---

## ⚡ Quick Setup (3 Steps)

### Step 1 — Install
```bash
pip install -r requirements.txt
```

### Step 2 — Config
`config.json` mein apna Telegram Bot Token daalo:
```json
{
  "telegram_token": "12345:ABC-DEF..."
}
```

> 💡 Telegram Bot Token kaise milega?
> @BotFather pe jao → /newbot → token copy karo

### Step 3 — Run
```bash
python bot.py
```

---

## 🔑 API Keys Bot Mein Kaise Daalen

Bot start hone ke baad, **bot ko private message karo**:

```
/setapi openai sk-proj-...
/setapi claude sk-ant-api03-...
/setapi gemini AIzaSy...
/setapi groq gsk_...
/setapi mistral ...
```

> ✅ Message automatically delete ho jayega (security ke liye)

---

## 📖 Commands

| Command | Kaam |
|---------|------|
| `/start` | Bot shuru karo |
| `/help` | Sab commands dekho |
| `/ask <sawaal>` | AI se seedha poocho |
| `/build <description>` | App/code banwao |
| `/setapi <provider> <key>` | API key set karo |
| `/status` | Current settings dekho |
| `/reset` | Chat history saaf karo |

---

## 📁 File Upload

Koi bhi code file ya ZIP bhejo + caption mein batao kya karna hai:

```
[File: myapp.py]
Caption: "Is code mein bugs dhundho aur fix karo"
```

```
[File: project.zip]
Caption: "Pura project review karo aur improve karo"
```

**Supported files:** .py .js .ts .jsx .tsx .html .css .java .cpp .c .go .rs .php .rb .swift .kt .sql .json .yaml .zip + more

---

## 🤖 Supported AI Providers

| Provider | Models |
|----------|--------|
| **OpenAI** | GPT-4o, GPT-4-turbo, GPT-3.5-turbo |
| **Claude** | claude-opus-4-5, claude-sonnet-4-5, claude-haiku-4-5 |
| **Gemini** | gemini-1.5-pro, gemini-1.5-flash |
| **Groq** | llama3-70b, mixtral-8x7b |
| **Mistral** | mistral-large, mistral-medium |
| **Together AI** | Llama-3-70b, Mixtral-8x22B |
| **Cohere** | command-r-plus, command-r |

---

## 🔄 Provider/Model Kaise Badlen

**Option 1 — Button se (aasaan):**
`/start` → "Provider Badlo" ya "Model Badlo" button dabaao

**Option 2 — Menu se:**
Bot mein `/start` → inline buttons use karo

---

## 🗂️ File Structure

```
telegram_ai_bot/
├── bot.py              ← Main bot script
├── config.json         ← API keys & settings
├── user_settings.json  ← Per-user settings (auto-created)
├── requirements.txt    ← Dependencies
└── README.md
```

---

## ☁️ Server Pe Deploy Karna (24/7 chalane ke liye)

### Option A — Screen (Linux)
```bash
screen -S aibot
python bot.py
# Ctrl+A, D to detach
```

### Option B — Systemd Service
```ini
[Unit]
Description=AI Telegram Bot

[Service]
WorkingDirectory=/path/to/telegram_ai_bot
ExecStart=/usr/bin/python3 bot.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### Option C — Railway/Render (Free hosting)
Environment variable `TELEGRAM_TOKEN` set karo, bot.py upload karo.

---

## 🔒 Security Tips

- `config.json` ko `.gitignore` mein daalo
- `/setapi` commands private chat mein bhejo
- API keys kabhi public share mat karo
