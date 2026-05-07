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

import openai
import anthropic
from google import genai as genai_client
import requests

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_FILE = "config.json"
USER_SETTINGS_FILE = "user_settings.json"

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

class ConfigManager:
    def __init__(self):
        self.config = self._load_config()
        self.user_settings = self._load_user_settings()

    def _load_config(self) -> dict:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                loaded = json.load(f)
                merged = DEFAULT_CONFIG.copy()
                merged.update(loaded)
                cfg = merged
        else:
            cfg = DEFAULT_CONFIG.copy()

        if os.environ.get("TELEGRAM_TOKEN"):
            cfg["telegram_token"] = os.environ["TELEGRAM_TOKEN"]

        env_map = {
            "openai":   "OPENAI_API_KEY",
            "claude":   "CLAUDE_API_KEY",
            "gemini":   "GEMINI_API_KEY",
            "groq":     "GROQ_API_KEY",
            "mistral":  "MISTRAL_API_KEY",
            "together": "TOGETHER_API_KEY",
            "cohere":   "COHERE_API_KEY",
        }
        for provider, env_key in env_map.items():
            val = os.environ.get(env_key, "")
            if val:
                cfg["apis"][provider]["key"] = val
                cfg["apis"][provider]["enabled"] = True

        if os.environ.get("DEFAULT_PROVIDER"):
            cfg["default_provider"] = os.environ["DEFAULT_PROVIDER"]
        if os.environ.get("DEFAULT_MODEL"):
            cfg["default_model"] = os.environ["DEFAULT_MODEL"]

        return cfg

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
            model=model, max_tokens=4096, system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text

    async def _gemini(self, key, model, prompt):
        client = genai_client.Client(api_key=key)
        resp = client.models.generate_content(model=model, contents=prompt)
        return resp.text

    async def _groq(self, key, model, prompt, system):
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": prompt}]}
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers=headers, json=payload, timeout=60)
        return r.json()["choices"][0]["message"]["content"]

    async def _mistral(self, key, model, prompt, system):
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": prompt}]}
        r = requests.post("https://api.mistral.ai/v1/chat/completions",
                          headers=headers, json=payload, timeout=60)
        return r.json()["choices"][0]["message"]["content"]

    async def _together(self, key, model, prompt, system):
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": prompt}], "max_tokens": 4096}
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

class FileProcessor:
    MAX_CHARS = 8000

    def extract_files(self, file_path: str) -> dict:
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

    def build_review_prompt(self, files: dict, user_request: str) -> str:
        prompt = f"User Request: {user_request}\n\n=== Uploaded Files ===\n\n"
        for fname, content in files.items():
            prompt += f"--- File: {fname} ---\n```\n{content}\n```\n\n"
        return prompt

file_processor = FileProcessor()

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
    rows = [[InlineKeyboardButton(m.split("/")[-1][:35], callback_data=f"set_model_{m}")] for m in models]
    rows.append([InlineKeyboardButton("◀ Back", callback_data="menu_main")])
    return InlineKeyboardMarkup(rows)

async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    provider = config_manager.get_provider(user.id)
    model = config_manager.get_model(user.id)
    text = (f"🤖 *Advanced AI Bot* mein aapka swagat hai!\n\n"
            f"👤 *User:* {user.first_name}\n🔌 *Provider:* `{provider}`\n🧠 *Model:* `{model}`\n\n"
            f"📁 Code/ZIP files bhejo → AI review karega!\n💬 Kuch bhi poocho ya app banwao!\n\n"
            f"⚡ Use `/help` for all commands.")
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_keyboard())

async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = ("📖 *BOT COMMANDS*\n\n"
            "`/ask <sawaal>` — AI se seedha poocho\n"
            "`/build <description>` — App/code banwao\n"
            "`/setapi <provider> <key>` — API key set karo\n"
            "`/status` — Current settings dekho\n"
            "`/reset` — Chat history reset karo\n\n"
            "*Providers:* openai, claude, gemini, groq, mistral, together, cohere\n\n"
            "*File Upload:* .py .js .ts .html .java .cpp .go .zip +more")
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def status_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    provider = config_manager.get_provider(uid)
    model = config_manager.get_model(uid)
    lines = [f"📊 *Bot Status*\n", f"🔌 Provider: `{provider}`", f"🧠 Model: `{model}`\n", "*API Keys:*"]
    for p, data in config_manager.config["apis"].items():
        key = data.get("key", "")
        status = "✅ Set" if key else "❌ Not Set"
        lines.append(f"• `{p}`: {status}")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

async def set_api_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = ctx.args
    if len(args) < 2:
        await update.message.reply_text("❌ Usage: `/setapi <provider> <api_key>`", parse_mode=ParseMode.MARKDOWN)
        return
    provider = args[0].lower()
    api_key = args[1].strip()
    if provider not in config_manager.config["apis"]:
        await update.message.reply_text(f"❌ Unknown provider: `{provider}`", parse_mode=ParseMode.MARKDOWN)
        return
    config_manager.config["apis"][provider]["key"] = api_key
    config_manager.config["apis"][provider]["enabled"] = True
    config_manager.save_config()
    try:
        await update.message.delete()
    except Exception:
        pass
    await ctx.bot.send_message(update.effective_chat.id,
        f"✅ `{provider.capitalize()}` API key saved!\n🔒 Message deleted for security.",
        parse_mode=ParseMode.MARKDOWN)

async def ask_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("❓ Usage: `/ask <your question>`", parse_mode=ParseMode.MARKDOWN)
        return
    question = " ".join(ctx.args)
    uid = update.effective_user.id
    msg = await update.message.reply_text("⏳ Soch raha hoon...")
    response = await ai_handler.query(config_manager.get_provider(uid), config_manager.get_model(uid), question)
    if len(response) > 4000:
        chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
        await msg.edit_text(chunks[0])
        for chunk in chunks[1:]:
            await update.message.reply_text(chunk)
    else:
        await msg.edit_text(response)

async def build_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("🔨 Usage: `/build <description>`", parse_mode=ParseMode.MARKDOWN)
        return
    desc = " ".join(ctx.args)
    uid = update.effective_user.id
    system = ("You are an expert software developer. Provide complete, working code with "
              "proper structure, comments, and setup instructions. Format code in markdown code blocks.")
    msg = await update.message.reply_text("🔨 App bana raha hoon...")
    response = await ai_handler.query(config_manager.get_provider(uid), config_manager.get_model(uid),
                                      f"Build this application: {desc}", system)
    if len(response) > 4000:
        chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
        await msg.edit_text(f"📦 *App Generated:*\n\n{chunks[0]}", parse_mode=ParseMode.MARKDOWN)
        for chunk in chunks[1:]:
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
    else:
        await msg.edit_text(f"📦 *App Generated:*\n\n{response}", parse_mode=ParseMode.MARKDOWN)

async def file_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    doc: Document = update.message.document
    uid = update.effective_user.id
    max_bytes = DEFAULT_CONFIG["max_file_size_mb"] * 1024 * 1024
    if doc.file_size > max_bytes:
        await update.message.reply_text(f"❌ File too large! Max: {DEFAULT_CONFIG['max_file_size_mb']}MB")
        return
    fname = doc.file_name or "unknown"
    ext = Path(fname).suffix.lower()
    if ext not in DEFAULT_CONFIG["allowed_extensions"]:
        await update.message.reply_text(f"❌ File type `{ext}` supported nahi hai.", parse_mode=ParseMode.MARKDOWN)
        return
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
        file_list = "\n".join(f"• `{f}`" for f in files.keys())
        await msg.edit_text(f"📁 *Files mili:*\n{file_list}\n\n⏳ AI analyze kar raha hai...", parse_mode=ParseMode.MARKDOWN)
        system = ("You are an expert code reviewer. Provide: 1) Code quality assessment, "
                  "2) Bugs found, 3) Security issues, 4) Optimization suggestions, 5) Fixed code if needed.")
        prompt = file_processor.build_review_prompt(files, user_request)
        response = await ai_handler.query(config_manager.get_provider(uid), config_manager.get_model(uid), prompt, system)
        full = f"🤖 *AI Analysis*\n\n{response}"
        if len(full) > 4000:
            chunks = [full[i:i+4000] for i in range(0, len(full), 4000)]
            await msg.edit_text(chunks[0], parse_mode=ParseMode.MARKDOWN)
            for chunk in chunks[1:]:
                await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
        else:
            await msg.edit_text(full, parse_mode=ParseMode.MARKDOWN)

async def text_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    uid = update.effective_user.id
    if not config_manager.get_user_setting(uid, "chat_mode", True):
        return
    history = config_manager.get_user_setting(uid, "history", [])
    history.append({"role": "user", "content": text})
    if len(history) > 10:
        history = history[-10:]
    msg = await update.message.reply_text("💭 Soch raha hoon...")
    history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history[:-1]])
    prompt = f"Conversation history:\n{history_text}\n\nUser: {text}" if history_text else text
    response = await ai_handler.query(config_manager.get_provider(uid), config_manager.get_model(uid), prompt)
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
    config_manager.set_user_setting(update.effective_user.id, "history", [])
    await update.message.reply_text("✅ Conversation history reset ho gaya!")

async def callback_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    uid = query.from_user.id

    back_btn = InlineKeyboardMarkup([[InlineKeyboardButton("◀ Back", callback_data="menu_main")]])

    if data == "menu_main":
        await query.edit_message_text("🏠 *Main Menu*", parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_keyboard())
    elif data == "menu_provider":
        await query.edit_message_text(f"🔌 *Provider Select Karo*\nCurrent: `{config_manager.get_provider(uid)}`",
                                      parse_mode=ParseMode.MARKDOWN, reply_markup=provider_keyboard())
    elif data == "menu_model":
        await query.edit_message_text(f"🧠 *Model Select Karo*\nProvider: `{config_manager.get_provider(uid)}`",
                                      parse_mode=ParseMode.MARKDOWN, reply_markup=model_keyboard(uid))
    elif data == "menu_status":
        lines = [f"📊 *Status*\n🔌 `{config_manager.get_provider(uid)}`\n🧠 `{config_manager.get_model(uid)}`\n\n*API Keys:*"]
        for p, d in config_manager.config["apis"].items():
            s = "✅" if d.get("key") else "❌"
            lines.append(f"{s} `{p}`")
        await query.edit_message_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN, reply_markup=back_btn)
    elif data == "menu_api":
        await query.edit_message_text(
            "🔑 *API Key Set Karo:*\n\n`/setapi openai sk-...`\n`/setapi claude sk-ant-...`\n"
            "`/setapi gemini AIza...`\n`/setapi groq gsk_...`\n\n⚠️ Message auto-delete hoga!",
            parse_mode=ParseMode.MARKDOWN, reply_markup=back_btn)
    elif data == "menu_chat":
        new_mode = not config_manager.get_user_setting(uid, "chat_mode", True)
        config_manager.set_user_setting(uid, "chat_mode", new_mode)
        await query.edit_message_text(f"💬 Chat Mode: *{'ON ✅' if new_mode else 'OFF ❌'}*",
                                      parse_mode=ParseMode.MARKDOWN, reply_markup=back_btn)
    elif data == "menu_help":
        await query.edit_message_text("❓ `/help` command use karo!", parse_mode=ParseMode.MARKDOWN, reply_markup=back_btn)
    elif data.startswith("set_provider_"):
        provider = data.replace("set_provider_", "")
        config_manager.set_user_setting(uid, "provider", provider)
        default_model = config_manager.config["models"].get(provider, ["gpt-4o"])[0]
        config_manager.set_user_setting(uid, "model", default_model)
        warn = "" if config_manager.is_provider_enabled(provider) else f"\n\n⚠️ `/setapi {provider} YOUR_KEY`"
        await query.edit_message_text(f"✅ Provider: `{provider}`\n🧠 Model: `{default_model}`{warn}",
                                      parse_mode=ParseMode.MARKDOWN, reply_markup=back_btn)
    elif data.startswith("set_model_"):
        model = data.replace("set_model_", "")
        config_manager.set_user_setting(uid, "model", model)
        await query.edit_message_text(f"✅ Model set: `{model}`", parse_mode=ParseMode.MARKDOWN, reply_markup=back_btn)

async def error_handler(update: object, ctx: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Exception: {ctx.error}", exc_info=ctx.error)

def main():
    token = config_manager.config.get("telegram_token", "")
    if not token or token == "YOUR_TELEGRAM_BOT_TOKEN":
        token = os.environ.get("TELEGRAM_TOKEN", "")
        if not token:
            print("❌ TELEGRAM_TOKEN environment variable set karo!")
            return

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("setapi", set_api_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("build", build_cmd))
    app.add_handler(CommandHandler("reset", reset_cmd))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.Document.ALL, file_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.add_error_handler(error_handler)

    async def post_init(app):
        await app.bot.set_my_commands([
            BotCommand("start", "Bot start karo"),
            BotCommand("help", "Help dekho"),
            BotCommand("ask", "AI se poocho"),
            BotCommand("build", "App banwao"),
            BotCommand("setapi", "API key set karo"),
            BotCommand("status", "Settings dekho"),
            BotCommand("reset", "History reset karo"),
        ])
    app.post_init = post_init

    print("🤖 Bot start ho raha hai...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()