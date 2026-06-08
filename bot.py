"""
Telegram bot with SQLite-backed persistent memory.
Survives Railway restarts when a Volume is mounted at /data.

Requirements:
    pip install python-telegram-bot openai

Environment variables (set in Railway):
    TELEGRAM_TOKEN   — your bot token from @BotFather
    OPENAI_API_KEY   — your OpenAI API key

Railway Volume setup (one-time):
    1. Railway dashboard → your service → Volumes → Add Volume
    2. Mount path: /data
    3. That's it. memory.db will live at /data/memory.db and survive
       every redeploy, restart, and infrastructure update.

    Without a Volume the file is inside the container and gets wiped
    on every redeploy — memories would be lost.
"""

import os
import sqlite3
import asyncio
import logging
from contextlib import contextmanager

from openai import AsyncOpenAI
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# ─── Config ──────────────────────────────────────────────────────────────────

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MODEL           = "gpt-4.1-mini"
MAX_HISTORY     = 40        # messages kept in conversation (excl. system prompt)

# /data is a Railway persistent Volume (survives redeploys).
# Falls back to local memory.db so the bot still works during local development.
DB_PATH = "/data/memory.db" if os.path.isdir("/data") else "memory.db"

SYSTEM_PROMPT = """You are a helpful accounting assistant for startups.
You have access to information the user has previously shared (their memories).
Use that context naturally when answering questions.
Be concise, accurate, and friendly."""

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ─── Database ────────────────────────────────────────────────────────────────

def init_db():
    """Create tables if they don't exist."""
    with get_db() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                role       TEXT    NOT NULL,   -- 'user' or 'assistant'
                content    TEXT    NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS memories (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                fact       TEXT    NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_conv_user   ON conversations(user_id);
            CREATE INDEX IF NOT EXISTS idx_memory_user ON memories(user_id);
        """)
    log.info("Database ready at %s", DB_PATH)


@contextmanager
def get_db():
    """Yield a SQLite connection that auto-commits or rolls back."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─── Memory helpers ──────────────────────────────────────────────────────────

def load_memories(user_id: int) -> list[str]:
    """Return all stored facts for a user."""
    with get_db() as db:
        rows = db.execute(
            "SELECT fact FROM memories WHERE user_id = ? ORDER BY created_at",
            (user_id,)
        ).fetchall()
    return [row["fact"] for row in rows]


def save_memory(user_id: int, fact: str):
    """Persist a new fact."""
    with get_db() as db:
        db.execute(
            "INSERT INTO memories (user_id, fact) VALUES (?, ?)",
            (user_id, fact.strip())
        )
    log.info("Memory saved for user %s: %s", user_id, fact.strip())


def clear_memories(user_id: int):
    with get_db() as db:
        db.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))


# ─── Conversation helpers ────────────────────────────────────────────────────

def load_history(user_id: int, limit: int = MAX_HISTORY) -> list[dict]:
    """Return the last `limit` messages as OpenAI message dicts."""
    with get_db() as db:
        rows = db.execute(
            """
            SELECT role, content FROM conversations
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit)
        ).fetchall()
    # Rows come back newest-first; reverse so oldest is first
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def save_message(user_id: int, role: str, content: str):
    """Append one message to the conversation log."""
    with get_db() as db:
        db.execute(
            "INSERT INTO conversations (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content)
        )


def clear_history(user_id: int):
    with get_db() as db:
        db.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))


# ─── Build the messages array for OpenAI ─────────────────────────────────────

def build_messages(user_id: int, new_user_text: str) -> list[dict]:
    """
    Construct the full messages array:
        1. System prompt (with injected memories)
        2. Recent conversation history
        3. The new user message
    """
    memories = load_memories(user_id)
    memory_block = ""
    if memories:
        facts = "\n".join(f"- {f}" for f in memories)
        memory_block = f"\n\nKnown facts about this user:\n{facts}"

    system = {"role": "system", "content": SYSTEM_PROMPT + memory_block}
    history = load_history(user_id)
    new_msg = {"role": "user", "content": new_user_text}

    return [system] + history + [new_msg]


# ─── OpenAI call ─────────────────────────────────────────────────────────────

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Per-user async locks (in-process only; fine because Railway runs one process)
_locks: dict[int, asyncio.Lock] = {}

def get_lock(user_id: int) -> asyncio.Lock:
    if user_id not in _locks:
        _locks[user_id] = asyncio.Lock()
    return _locks[user_id]


async def ask_openai(user_id: int, user_text: str) -> str:
    messages = build_messages(user_id, user_text)
    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=1000,
    )
    return response.choices[0].message.content.strip()


# ─── Telegram handlers ────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hi! I'm your accounting assistant.\n\n"
        "💡 Tip: use `remember: <fact>` to save something permanently.\n"
        "Example: `remember: My startup is Acme AI`\n\n"
        "Commands:\n"
        "/memories — show what I know about you\n"
        "/forget   — delete all your saved facts\n"
        "/reset    — clear conversation history\n"
        "/help     — show this message"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, context)


async def cmd_memories(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    facts = load_memories(user_id)
    if not facts:
        await update.message.reply_text("I don't have any saved facts about you yet.")
    else:
        lines = "\n".join(f"• {f}" for f in facts)
        await update.message.reply_text(f"📋 What I know about you:\n\n{lines}")


async def cmd_forget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    clear_memories(user_id)
    await update.message.reply_text("✅ All your saved facts have been deleted.")


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    clear_history(user_id)
    await update.message.reply_text("✅ Conversation history cleared. Fresh start!")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id   = update.message.from_user.id
    user_text = update.message.text.strip()

    # ── Handle remember: command ──────────────────────────────────────────────
    if user_text.lower().startswith("remember:"):
        fact = user_text[len("remember:"):].strip()
        if fact:
            save_memory(user_id, fact)
            await update.message.reply_text(f"✅ Got it, I'll remember:\n_{fact}_", parse_mode="Markdown")
        else:
            await update.message.reply_text("Please add a fact after `remember:`, e.g.:\n`remember: My startup is Acme AI`")
        return

    # ── Normal conversation ───────────────────────────────────────────────────
    async with get_lock(user_id):
        await context.bot.send_chat_action(update.effective_chat.id, "typing")

        try:
            reply = await ask_openai(user_id, user_text)
        except Exception as e:
            log.error("OpenAI error for user %s: %s", user_id, e)
            await update.message.reply_text("Sorry, something went wrong. Please try again.")
            return

        # Persist both sides of the exchange
        save_message(user_id, "user",      user_text)
        save_message(user_id, "assistant", reply)

    await update.message.reply_text(reply)


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    init_db()

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .build()
    )

    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("help",     cmd_help))
    app.add_handler(CommandHandler("memories", cmd_memories))
    app.add_handler(CommandHandler("forget",   cmd_forget))
    app.add_handler(CommandHandler("reset",    cmd_reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("Bot started.")
    app.run_polling()


if __name__ == "__main__":
    main()
