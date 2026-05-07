#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════╗
║         ADVANCED AI TELEGRAM BOT - Multi Model       ║
║   Supports: OpenAI, Claude, Gemini, Groq, Mistral   ║
╚══════════════════════════════════════════════════════╝
"""

import os
import re
import json
import zipfile
import tempfile
import asyncio
import logging
from pathlib import Path
from typing import Optional

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    BotCommand, Document
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)
from telegram.constants import ParseMode

# AI Provider Imports
import openai
import anthropic
import google.generativeai as genai
import requests

# ─── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─── Config File Path ──────────────────────────────────────────────────────────
CONFIG_FILE = "config.json"
USER_SETTINGS_FILE = "user_settings.json"

# ─── Default Config ────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "telegram_token": "YOUR_TELEGRAM_BOT_TOKEN",
    "apis": {
        "openai":    {"key": "", "enabled": False},
        "claude":    {"key": "", "enabled": False},
        "gemini":    {"key": "", "enabled": False},
        "groq":      {"key": "", "enabled": False},
        "mistral":   {"key": "", "enabled": False},
        "together":  {"key": "", "enabled": False},
        "cohere":    {"key": "", "enabled": False},
    },
    "models": {
        "openai":   ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o-mini"],
        "claude":   ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"],
        "gemini":   ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
        "groq":     ["llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
        "mistral":  ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"],
        "together": ["meta-llama/Llama-3-70b-chat-hf", "mistralai/Mixtral-8x22B-Instruct-v0.1"],
        "cohere":   ["command-r-plus", "command-r", "command"],
    },
    "default_provider": "openai",
    "default_model": "gpt-4o",
    "max_file_size_mb": 10,
    "allowed_extensions": [".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css",
                           ".java", ".cpp", ".c", ".cs", ".go", ".rs", ".php",
                           ".rb", ".swift", ".kt", ".dart", ".r", ".sql",
                           ".json", ".yaml", ".yml", ".toml", ".md", ".txt",
                           ".zip", ".sh", ".bash", ".env.example"]
}

# ─── Config Manager ────────────────────────────────────────────────────────────
class ConfigManager:
    def __init__(self):
        self.config = self._load_config()
        self.user_settings = self._load_user_settings()

    def _load_config(self) -> dict:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                loaded = json.load(f)
                # Merge with defaults for new keys
                merged = DEFAULT_CONFIG.copy()
                merged.update(loaded)
                return merged
        self._save_json(CONFIG_FILE, DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

    def _load_user_settings(self) -> dict:
        if os.path.exists(USER_SETTINGS_FILE):
            with open(USER_SETTINGS_FILE, "r") as f:
                return json.load(f)
        return {}

    def _save_json(self, path: str, data: dict):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def save_config(self):
        self._save_json(CONFIG_FILE, self.config)

    def get_user_setting(self, user_id: int, key: str, default=None):
        uid = str(user_id)
        return self.user_settings.get(uid, {}).get(key, default)

    def set_user_setting(self, user_id: int, key: str, value):
        uid = str(user_id)
        if uid not in self.user_settings:
            self.user_settings[uid] = {}
        self.user_settings[uid][key] = value
        self._save_json(USER_SETTINGS_FILE, self.user_settings)

    def get_provider(self, user_id: int) -> str:
        return self.get_user_setting(user_id, "provider", self.config["default_provider"])

    def get_model(self, user_id: int) -> str:
        provider = self.get_provider(user_id)
        default_model = self.config["models"].get(provider, ["gpt-4o"])[0]
        return self.get_user_setting(user_id, "model", default_model)

    def get_api_key(self, provider: str) -> str:
        return self.config["apis"].get(provider, {}).get("key", "")

    def is_provider_enabled(self, provider: str) -> bool:
        api_data = self.config["apis"].get(provider, {})
        return api_data.get("enabled", False) and bool(api_data.get("key", ""))

config_manager = ConfigManager()

# ─── AI Provider Handler ───────────────────────────────────────────────────────
class AIHandler:
    def __init__(self, cfg: ConfigManager):
        self.cfg = cfg

    async def query(self, provider: str, model: str, prompt: str,
                    system: str = "You are a helpful AI assistant.") -> str:
        key = self.cfg.get_api_key(provider)
        if not key:
            return f"❌ `{provider}` ka API key set nahi hai. `/setapi {provider} YOUR_KEY` use karo."

        try:
            if provider == "openai":
                return await self._openai(key, model, prompt, system)
            elif provider == "claude":
                return await self._claude(key, model, prompt, system)
            elif provider == "gemini":
                return await self._gemini(key, model, prompt)
            elif provider == "groq":
                return await self._groq(key, model, prompt, system)
            elif provider == "mistral":
                return await self._mistral(key, model, prompt, system)
            elif provider == "together":
                return await self._together(key, model, prompt, system)
            elif provider == "cohere":
                return await self._cohere(key, model, prompt)
            else:
                return "❌ Unknown provider."
        except Exception as e:
            logger.error(f"AI Error [{provider}]: {e}")
            return f"❌ Error: `{str(e)}`"

    async def _openai(self, key, model, prompt, system):
        client = openai.OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=0.7
        )
        return resp.choices[0].message.content

    async def _claude(self, key, model, prompt, system):
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text

    async def _gemini(self, key, model, prompt):
        genai.configure(api_key=key)
        m = genai.GenerativeModel(model)
        resp = m.generate_content(prompt)
        return resp.text

    async def _groq(self, key, model, prompt, system):
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}]
        }
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers=headers, json=payload, timeout=60)
        return r.json()["choices"][0]["message"]["content"]

    async def _mistral(self, key, model, prompt, system):
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}]
        }
        r = requests.post("https://api.mistral.ai/v1/chat/completions",
                          headers=headers, json=payload, timeout=60)
        return r.json()["choices"][0]["message"]["content"]

    async def _together(self, key, model, prompt, system):
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "max_tokens": 4096
        }
        r = requests.post("https://api.together.xyz/v1/chat/completions",
                          headers=headers, json=payload, timeout=60)
        return r.json()["choices"][0]["message"]["content"]

    async def _cohere(self, key, model, prompt):
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {"model": model, "message": prompt, "max_tokens": 4096}
        r = requests.post("https://api.cohere.ai/v1/chat",
                          headers=headers, json=payload, timeout=60)
        return r.json()["text"]

ai_handler = AIHandler(config_manager)

# ─── File Processor ────────────────────────────────────────────────────────────
class FileProcessor:
    MAX_CHARS = 8000  # chars per file to avoid token overflow

    def extract_files(self, file_path: str) -> dict[str, str]:
        """Returns {filename: content} dict from file or zip"""
        results = {}
        ext = Path(file_path).suffix.lower()

        if ext == ".zip":
            with zipfile.ZipFile(file_path, "r") as zf:
                for name in zf.namelist():
                    if any(name.endswith(e) for e in DEFAULT_CONFIG["allowed_extensions"]):
                        try:
                            content = zf.read(name).decode("utf-8", errors="replace")
                            results[name] = content[:self.MAX_CHARS]
                        except Exception:
                            pass
        else:
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    results[Path(file_path).name] = f.read()[:self.MAX_CHARS]
            except Exception:
                pass

        return results

    def build_review_prompt(self, files: dict[str, str], user_request: str) -> str:
        prompt = f"User Request: {user_request}\n\n"
        prompt += "=== Uploaded Files ===\n\n"
        for fname, content in files.items():
            prompt += f"--- File: {fname} ---\n```\n{content}\n```\n\n"
        return prompt

file_processor = FileProcessor()

# ─── Keyboards ─────────────────────────────────────────────────────────────────
def main_menu_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🤖 Provider Badlo", callback_data="menu_provider"),
         InlineKeyboardButton("🧠 Model Badlo", callback_data="menu_model")],
        [InlineKeyboardButton("🔑 API Keys", callback_data="menu_api"),
         InlineKeyboardButton("📊 Status", callback_data="menu_status")],
        [InlineKeyboardButton("💬 Chat Mode", callback_data="menu_chat"),
         InlineKeyboardButton("❓ Help", callback_data="menu_help")],
    ])

def provider_keyboard():
    providers = list(DEFAULT_CONFIG["apis"].keys())
    rows = []
    for i in range(0, len(providers), 2):
        row = [InlineKeyboardButton(p.capitalize(), callback_data=f"set_provider_{p}")
               for p in providers[i:i+2]]
        rows.append(row)
    rows.append([InlineKeyboardButton("◀ Back", callback_data="menu_main")])
    return InlineKeyboardMarkup(rows)

def model_keyboard(user_id: int):
    provider = config_manager.get_provider(user_id)
    models = config_manager.config["models"].get(provider, [])
    rows = []
    for i in range(0, len(models), 1):
        m = models[i]
        short = m.split("/")[-1][:35]
        rows.append([InlineKeyboardButton(short, callback_data=f"set_model_{m}")])
    rows.append([InlineKeyboardButton("◀ Back", callback_data="menu_main")])
    return InlineKeyboardMarkup(rows)

# ─── Command Handlers ──────────────────────────────────────────────────────────
async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    provider = config_manager.get_provider(user.id)
    model = config_manager.get_model(user.id)

    text = (
        f"🤖 *Advanced AI Bot* mein aapka swagat hai!\n\n"
        f"👤 *User:* {user.first_name}\n"
        f"🔌 *Provider:* `{provider}`\n"
        f"🧠 *Model:* `{model}`\n\n"
        f"📁 *File Upload:* Code, ZIP files bhejo → AI review karega!\n"
        f"💬 *Chat:* Kuch bhi poocho ya app banwao!\n\n"
        f"⚡ Use `/help` for all commands."
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN,
                                    reply_markup=main_menu_keyboard())

async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = """
📖 *BOT COMMANDS*

*⚙️ Setup*
`/setapi <provider> <key>` — API key set karo
`/provider` — Provider badlo (menu)
`/model` — Model badlo (menu)
`/status` — Current settings dekho

*📁 File Commands*
`/review` — File send karo, AI review karega
`/build <description>` — App/code banwao
`/explain` — Code explain karwao
`/fix` — Code fix karwao
`/optimize` — Code optimize karwao

*💬 Chat*
`/ask <question>` — AI se seedha poocho
`/reset` — Conversation reset karo

*🤖 Providers Supported*
• OpenAI (GPT-4o, GPT-4-turbo...)
• Claude (Opus, Sonnet, Haiku)
• Gemini (1.5-Pro, Flash...)
• Groq (Llama3, Mixtral...)
• Mistral (Large, Medium...)
• Together AI
• Cohere

*📁 Supported Files*
.py .js .ts .html .css .java .cpp .go .rs
.php .rb .swift .kt .sql .json .yaml .zip +more
"""
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def status_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    provider = config_manager.get_provider(uid)
    model = config_manager.get_model(uid)

    lines = [f"📊 *Bot Status*\n",
             f"🔌 *Active Provider:* `{provider}`",
             f"🧠 *Active Model:* `{model}`\n",
             f"*API Keys Status:*"]

    for p, data in config_manager.config["apis"].items():
        key = data.get("key", "")
        status = "✅ Set" if key and key != "YOUR_KEY_HERE" else "❌ Not Set"
        lines.append(f"• `{p}`: {status}")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

async def set_api_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = ctx.args
    if len(args) < 2:
        await update.message.reply_text(
            "❌ Usage: `/setapi <provider> <api_key>`\n\n"
            "Providers: openai, claude, gemini, groq, mistral, together, cohere",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    provider = args[0].lower()
    api_key = args[1].strip()

    if provider not in config_manager.config["apis"]:
        await update.message.reply_text(f"❌ Unknown provider: `{provider}`",
                                        parse_mode=ParseMode.MARKDOWN)
        return

    config_manager.config["apis"][provider]["key"] = api_key
    config_manager.config["apis"][provider]["enabled"] = True
    config_manager.save_config()

    # Delete the message for security
    try:
        await update.message.delete()
    except Exception:
        pass

    await ctx.bot.send_message(
        update.effective_chat.id,
        f"✅ `{provider.capitalize()}` API key saved!\n🔒 Message deleted for security.",
        parse_mode=ParseMode.MARKDOWN
    )

async def ask_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("❓ Usage: `/ask <your question>`",
                                        parse_mode=ParseMode.MARKDOWN)
        return

    question = " ".join(ctx.args)
    uid = update.effective_user.id
    provider = config_manager.get_provider(uid)
    model = config_manager.get_model(uid)

    msg = await update.message.reply_text("⏳ Soch raha hoon...")
    response = await ai_handler.query(provider, model, question)

    # Split long responses
    if len(response) > 4000:
        chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
        await msg.edit_text(chunks[0])
        for chunk in chunks[1:]:
            await update.message.reply_text(chunk)
    else:
        await msg.edit_text(response)

async def build_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text(
            "🔨 Usage: `/build <description>`\n\nExample:\n`/build ek flask todo app banao with sqlite`",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    desc = " ".join(ctx.args)
    uid = update.effective_user.id
    provider = config_manager.get_provider(uid)
    model = config_manager.get_model(uid)

    system = (
        "You are an expert software developer. When asked to build an application, "
        "provide complete, working code with proper structure. Include all necessary files, "
        "clear comments, and setup instructions. Format code in markdown code blocks."
    )

    msg = await update.message.reply_text("🔨 App bana raha hoon...")
    response = await ai_handler.query(provider, model, f"Build this application: {desc}", system)

    if len(response) > 4000:
        chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
        await msg.edit_text(f"📦 *App Generated:*\n\n{chunks[0]}", parse_mode=ParseMode.MARKDOWN)
        for chunk in chunks[1:]:
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
    else:
        await msg.edit_text(f"📦 *App Generated:*\n\n{response}", parse_mode=ParseMode.MARKDOWN)

# ─── File Handler ──────────────────────────────────────────────────────────────
async def file_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    doc: Document = update.message.document
    uid = update.effective_user.id

    # Size check
    max_bytes = DEFAULT_CONFIG["max_file_size_mb"] * 1024 * 1024
    if doc.file_size > max_bytes:
        await update.message.reply_text(
            f"❌ File too large! Max size: {DEFAULT_CONFIG['max_file_size_mb']}MB"
        )
        return

    # Extension check
    fname = doc.file_name or "unknown"
    ext = Path(fname).suffix.lower()
    if ext not in DEFAULT_CONFIG["allowed_extensions"]:
        await update.message.reply_text(
            f"❌ File type `{ext}` supported nahi hai.\n\n"
            f"Supported: {', '.join(DEFAULT_CONFIG['allowed_extensions'][:15])} ...",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    # Caption = user's instruction
    user_request = update.message.caption or "Is code ko review karo aur improve karo."

    msg = await update.message.reply_text("📥 File download ho rahi hai...")

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, fname)
        tg_file = await ctx.bot.get_file(doc.file_id)
        await tg_file.download_to_drive(file_path)

        await msg.edit_text("🔍 Code analyze ho raha hai...")

        files = file_processor.extract_files(file_path)

        if not files:
            await msg.edit_text("❌ File mein readable code nahi mila.")
            return

        # Build prompt
        file_list = "\n".join(f"• `{f}`" for f in files.keys())
        await msg.edit_text(
            f"📁 *Files mili:*\n{file_list}\n\n⏳ AI se analysis kar raha hoon...",
            parse_mode=ParseMode.MARKDOWN
        )

        provider = config_manager.get_provider(uid)
        model = config_manager.get_model(uid)

        system = (
            "You are an expert code reviewer and developer. Analyze the provided code thoroughly. "
            "Provide: 1) Code quality assessment, 2) Bugs and issues found, "
            "3) Security vulnerabilities, 4) Optimization suggestions, "
            "5) Improved/fixed code if requested. Be specific and actionable."
        )

        prompt = file_processor.build_review_prompt(files, user_request)
        response = await ai_handler.query(provider, model, prompt, system)

        # Send response
        header = f"🤖 *AI Analysis* ({provider}/{model.split('/')[-1]})\n\n"
        full = header + response

        if len(full) > 4000:
            chunks = [full[i:i+4000] for i in range(0, len(full), 4000)]
            await msg.edit_text(chunks[0], parse_mode=ParseMode.MARKDOWN)
            for chunk in chunks[1:]:
                await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
        else:
            await msg.edit_text(full, parse_mode=ParseMode.MARKDOWN)

# ─── Text Message Handler ──────────────────────────────────────────────────────
async def text_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    uid = update.effective_user.id

    # Check if in chat mode
    chat_mode = config_manager.get_user_setting(uid, "chat_mode", True)
    if not chat_mode:
        return

    provider = config_manager.get_provider(uid)
    model = config_manager.get_model(uid)

    # Get conversation history
    history = config_manager.get_user_setting(uid, "history", [])
    history.append({"role": "user", "content": text})

    # Keep last 10 messages
    if len(history) > 10:
        history = history[-10:]

    msg = await update.message.reply_text("💭 Soch raha hoon...")

    # For conversation history, build prompt
    history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history[:-1]])
    prompt = f"Conversation history:\n{history_text}\n\nUser: {text}" if history_text else text

    response = await ai_handler.query(provider, model, prompt)
    history.append({"role": "assistant", "content": response})

    config_manager.set_user_setting(uid, "history", history[-10:])

    if len(response) > 4000:
        chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
        await msg.edit_text(chunks[0])
        for chunk in chunks[1:]:
            await update.message.reply_text(chunk)
    else:
        await msg.edit_text(response)

async def reset_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    config_manager.set_user_setting(uid, "history", [])
    await update.message.reply_text("✅ Conversation history reset ho gaya!")

# ─── Callback Query Handler ────────────────────────────────────────────────────
async def callback_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    uid = query.from_user.id

    if data == "menu_main":
        await query.edit_message_text("🏠 *Main Menu*", parse_mode=ParseMode.MARKDOWN,
                                      reply_markup=main_menu_keyboard())

    elif data == "menu_provider":
        current = config_manager.get_provider(uid)
        await query.edit_message_text(
            f"🔌 *Provider Select Karo*\nCurrent: `{current}`",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=provider_keyboard()
        )

    elif data == "menu_model":
        provider = config_manager.get_provider(uid)
        current_model = config_manager.get_model(uid)
        await query.edit_message_text(
            f"🧠 *Model Select Karo*\nProvider: `{provider}`\nCurrent: `{current_model}`",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=model_keyboard(uid)
        )

    elif data == "menu_status":
        provider = config_manager.get_provider(uid)
        model = config_manager.get_model(uid)
        lines = [f"📊 *Status*\n",
                 f"🔌 Provider: `{provider}`",
                 f"🧠 Model: `{model}`\n",
                 "*API Keys:*"]
        for p, d in config_manager.config["apis"].items():
            k = d.get("key", "")
            s = "✅" if k and k not in ("", "YOUR_KEY_HERE") else "❌"
            lines.append(f"{s} `{p}`")
        await query.edit_message_text(
            "\n".join(lines), parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("◀ Back", callback_data="menu_main")
            ]])
        )

    elif data == "menu_api":
        text = (
            "🔑 *API Key Set Karne Ka Tarika:*\n\n"
            "Bot mein private message karo:\n"
            "`/setapi openai sk-...`\n"
            "`/setapi claude sk-ant-...`\n"
            "`/setapi gemini AIza...`\n"
            "`/setapi groq gsk_...`\n"
            "`/setapi mistral ...`\n\n"
            "⚠️ Message automatically delete ho jayega!"
        )
        await query.edit_message_text(
            text, parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("◀ Back", callback_data="menu_main")
            ]])
        )

    elif data == "menu_chat":
        chat_mode = config_manager.get_user_setting(uid, "chat_mode", True)
        new_mode = not chat_mode
        config_manager.set_user_setting(uid, "chat_mode", new_mode)
        status = "ON ✅" if new_mode else "OFF ❌"
        await query.edit_message_text(
            f"💬 Chat Mode: *{status}*\n\nAb message bhejo to AI reply karega.",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("◀ Back", callback_data="menu_main")
            ]])
        )

    elif data == "menu_help":
        await query.edit_message_text(
            "❓ `/help` command use karo full help ke liye!",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("◀ Back", callback_data="menu_main")
            ]])
        )

    elif data.startswith("set_provider_"):
        provider = data.replace("set_provider_", "")
        config_manager.set_user_setting(uid, "provider", provider)
        # Reset model to provider default
        default_model = config_manager.config["models"].get(provider, ["gpt-4o"])[0]
        config_manager.set_user_setting(uid, "model", default_model)
        is_set = config_manager.is_provider_enabled(provider)
        key_warn = "" if is_set else f"\n\n⚠️ API key set karo: `/setapi {provider} YOUR_KEY`"
        await query.edit_message_text(
            f"✅ Provider: `{provider}`\n🧠 Model: `{default_model}`{key_warn}",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("◀ Back", callback_data="menu_main")
            ]])
        )

    elif data.startswith("set_model_"):
        model = data.replace("set_model_", "")
        config_manager.set_user_setting(uid, "model", model)
        await query.edit_message_text(
            f"✅ Model set: `{model}`",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("◀ Back", callback_data="menu_main")
            ]])
        )

# ─── Error Handler ─────────────────────────────────────────────────────────────
async def error_handler(update: object, ctx: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Exception: {ctx.error}", exc_info=ctx.error)

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    token = config_manager.config.get("telegram_token", "")
    if not token or token == "YOUR_TELEGRAM_BOT_TOKEN":
        print("❌ config.json mein telegram_token set karo!")
        print("   Ya TOKEN=... environment variable use karo")
        token = os.environ.get("TELEGRAM_TOKEN", "")
        if not token:
            return

    app = Application.builder().token(token).build()

    # Commands
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("setapi", set_api_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("build", build_cmd))
    app.add_handler(CommandHandler("reset", reset_cmd))
    app.add_handler(CommandHandler("provider", lambda u, c: callback_handler(
        type("Q", (), {"answer": lambda: None, "data": "menu_provider",
                       "from_user": u.effective_user,
                       "edit_message_text": u.message.reply_text})(), c)))

    # Callbacks
    app.add_handler(CallbackQueryHandler(callback_handler))

    # File handler
    app.add_handler(MessageHandler(filters.Document.ALL, file_handler))

    # Text handler
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    # Error handler
    app.add_error_handler(error_handler)

    # Set commands
    async def post_init(app):
        await app.bot.set_my_commands([
            BotCommand("start", "Bot start karo"),
            BotCommand("help", "Help dekho"),
            BotCommand("ask", "AI se seedha poocho"),
            BotCommand("build", "App/code banwao"),
            BotCommand("setapi", "API key set karo"),
            BotCommand("status", "Settings dekho"),
            BotCommand("reset", "Chat history reset karo"),
        ])

    app.post_init = post_init

    print("🤖 Bot start ho raha hai...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
