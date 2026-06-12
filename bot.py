"""
Georgian Financial Assistant Telegram Bot
- SQLite persistent memory (survives Railway restarts)
- Smart onboarding: detailed (13 questions) or quick (4 questions)
- /switch command to change mode without losing data
- Auto-saves business facts from conversation
- 50-message chat history
- OpenAI Responses API with File Search (Vector Store)

Environment variables:
    TELEGRAM_TOKEN   — BotFather token
    OPENAI_API_KEY   — OpenAI API key
    VECTOR_STORE_ID  — your vector store ID
"""

import os
import sqlite3
import asyncio
import logging
from contextlib import contextmanager

from openai import AsyncOpenAI
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

# ─── Config ───────────────────────────────────────────────────────────────────

TELEGRAM_TOKEN  = os.environ["TELEGRAM_TOKEN"]
OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
VECTOR_STORE_ID = os.environ.get("VECTOR_STORE_ID", "")

MODEL       = "gpt-4.1-mini"
MAX_HISTORY = 50
DB_PATH     = "/data/memory.db" if os.path.isdir("/data") else "memory.db"

# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Georgian accounting and financial AI assistant
with 20+ years of experience. You help startups and small businesses with:
- Tax declarations and obligations under Georgian law
- Choosing and optimizing tax regimes
- Financial planning and cash flow models
- Payroll, HR-related taxes, and salary calculations
- VAT registration and reporting
- Investor and bank reporting

You have access to uploaded accounting and financial documents via file search.
You also have access to saved facts about this specific user's business.
Use both sources to give accurate, personalized advice.

Always respond in the same language the user writes in.
Be professional, warm, and concise. Never be vague — give specific, actionable answers.
If a concept is unclear to the user and they ask for clarification, explain it
in very simple terms with a concrete real-life example.

## MEMORY MANAGEMENT
When the user shares business facts, extract and save them by adding at the
END of your response (these lines are hidden from display):

SAVE: [short clear fact in English]

Examples:
SAVE: company name — Acme AI
SAVE: legal form — LLC (შპს)
SAVE: tax regime — small business 1%
SAVE: monthly revenue — 15,000 GEL
SAVE: employees — 3, all on official payroll
SAVE: VAT registered — no
SAVE: last declaration — Q3 2024
SAVE: main expenses — salaries, office rent, marketing

If the user CORRECTS a previously saved fact:
UPDATE: [old fact keyword] → [new fact]

## ⚠️ CRITICAL ONBOARDING RULES — MUST FOLLOW EXACTLY
RULE 1: Send EXACTLY ONE question per message. ONE. Never two. Never three.
RULE 2: Do NOT combine questions with "და", "ასევე", "გარდა ამისა", "and", "also".
RULE 3: Count the "?" marks in your message before sending. If there are 2 or more — rewrite.
RULE 4: After the user answers, write SAVE: for that fact, then ask the NEXT single question.
RULE 5: If user asks for clarification mid-onboarding, answer it, then ask the SAME question again.
RULE 6: Short questions only. Maximum 2 sentences per question message.
RULE 7: End your last onboarding message with: ONBOARDING_COMPLETE

## DETAILED ONBOARDING (13 questions, one at a time)
Q1:  რა ჰქვია შენს კომპანიას და რა სფეროში მუშაობს?
Q2:  სამართლებრივი ფორმა რა არის — შპს, ინდ.მეწარმე, არარეგისტრირებული?
Q3:  რომელ საგადასახადო რეჟიმზე ხარ — მცირე ბიზნესი 1%, სტანდარტული 15%, ვირტუალური ზონა, სხვა?
Q4:  დღგ-ს გადამხდელი ხარ?
Q5:  ყოველთვიური საშუალო შემოსავალი რამდენია?
Q6:  შემოსავალი როგორ მოდის — მომსახურება, პროდუქტი, თუ სააბონენტო?
Q7:  კლიენტები ქართველები არიან, უცხოელები, თუ ორივე?
Q8:  თანამშრომლები გყავს?
Q9:  ძირითადი ხარჯები რა კატეგორიებშია?
Q10: ბუღალტერი გყავს თუ თავად აწარმოებ აღრიცხვას?
Q11: ბოლო საგადასახადო დეკლარაცია წარდგენილი გაქვს?
Q12: საბანკო სესხი, ინვესტიცია ან გრანტი გაქვს?
Q13: ახლა ყველაზე მწვავე ფინანსური კითხვა ან პრობლემა რა არის?

## QUICK ONBOARDING (4 questions, one at a time)
Q1: ბიზნესი რეგისტრირებულია და რა ფორმით — შპს, ინდ.მეწარმე, არარეგისტრირებული?
Q2: ყოველთვიური ბრუნვა დაახლოებით რამდენია?
Q3: თანამშრომლები გყავს?
Q4: რა გჭირდება ახლა — გადასახადები, დეკლარაცია, ფინანსური გეგმა, სხვა?
"""

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ─── Database ─────────────────────────────────────────────────────────────────

def init_db():
    with get_db() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                role       TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS user_state (
                user_id    INTEGER PRIMARY KEY,
                onboarding TEXT    DEFAULT 'none',
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id);
        """)

        # Rebuild memories table with correct schema
        db.execute("""
            CREATE TABLE IF NOT EXISTS memories_new (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                mem_key    TEXT    NOT NULL,
                fact       TEXT    NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, mem_key)
            )
        """)
        try:
            db.execute("""
                INSERT OR IGNORE INTO memories_new (user_id, mem_key, fact, created_at)
                SELECT user_id,
                       COALESCE(NULLIF(TRIM(COALESCE(key,'')), ''), SUBSTR(fact, 1, 40)),
                       fact, created_at
                FROM memories
            """)
            db.execute("DROP TABLE memories")
            db.execute("ALTER TABLE memories_new RENAME TO memories")
            log.info("Migration: memories table rebuilt")
        except Exception:
            try:
                db.execute("ALTER TABLE memories_new RENAME TO memories")
            except Exception:
                pass

        db.execute("CREATE INDEX IF NOT EXISTS idx_memory_user ON memories(user_id)")

    log.info("DB ready at %s", DB_PATH)


@contextmanager
def get_db():
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


# ─── State helpers ────────────────────────────────────────────────────────────

def get_onboarding_state(user_id: int) -> str:
    with get_db() as db:
        row = db.execute(
            "SELECT onboarding FROM user_state WHERE user_id = ?", (user_id,)
        ).fetchone()
    return row["onboarding"] if row else "none"


def set_onboarding_state(user_id: int, state: str):
    with get_db() as db:
        db.execute(
            """INSERT INTO user_state(user_id, onboarding, updated_at)
               VALUES(?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(user_id) DO UPDATE SET
                   onboarding=excluded.onboarding,
                   updated_at=excluded.updated_at""",
            (user_id, state)
        )


# ─── Memory helpers ───────────────────────────────────────────────────────────

def load_memories(user_id: int) -> list[str]:
    with get_db() as db:
        rows = db.execute(
            "SELECT fact FROM memories WHERE user_id = ? ORDER BY created_at",
            (user_id,)
        ).fetchall()
    return [row["fact"] for row in rows]


def save_memory(user_id: int, fact: str):
    fact    = fact.strip()
    mem_key = fact.split(" — ")[0].strip().lower() if " — " in fact else fact[:40].lower()
    with get_db() as db:
        db.execute("DELETE FROM memories WHERE user_id = ? AND mem_key = ?", (user_id, mem_key))
        db.execute("INSERT INTO memories(user_id, mem_key, fact) VALUES(?, ?, ?)", (user_id, mem_key, fact))
    log.info("Memory saved [%s]: %s", user_id, fact)


def update_memory(user_id: int, old_keyword: str, new_fact: str):
    new_fact = new_fact.strip()
    old_key  = old_keyword.strip().lower()
    new_key  = new_fact.split(" — ")[0].strip().lower() if " — " in new_fact else old_key
    with get_db() as db:
        db.execute("DELETE FROM memories WHERE user_id = ? AND mem_key = ?", (user_id, old_key))
        db.execute("INSERT INTO memories(user_id, mem_key, fact) VALUES(?, ?, ?)", (user_id, new_key, new_fact))
    log.info("Memory updated [%s]: %s -> %s", user_id, old_keyword, new_fact)


def clear_memories(user_id: int):
    with get_db() as db:
        db.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))


# ─── Conversation helpers ─────────────────────────────────────────────────────

def load_history(user_id: int) -> list[dict]:
    with get_db() as db:
        rows = db.execute(
            """SELECT role, content FROM conversations
               WHERE user_id = ? ORDER BY created_at DESC LIMIT ?""",
            (user_id, MAX_HISTORY)
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def save_message(user_id: int, role: str, content: str):
    with get_db() as db:
        db.execute(
            "INSERT INTO conversations(user_id, role, content) VALUES(?, ?, ?)",
            (user_id, role, content)
        )
        db.execute(
            """DELETE FROM conversations WHERE user_id = ? AND id NOT IN (
                SELECT id FROM conversations WHERE user_id = ?
                ORDER BY created_at DESC LIMIT ?
            )""",
            (user_id, user_id, MAX_HISTORY)
        )


def clear_history(user_id: int):
    with get_db() as db:
        db.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))


# ─── Parse AI response ────────────────────────────────────────────────────────

def extract_and_clean(user_id: int, raw_text: str) -> str:
    lines       = raw_text.split("\n")
    clean_lines = []
    for line in lines:
        s = line.strip()
        if s.startswith("SAVE:"):
            fact = s[5:].strip()
            if fact:
                save_memory(user_id, fact)
        elif s.startswith("UPDATE:"):
            payload = s[7:].strip()
            for sep in (" → ", " -> "):
                if sep in payload:
                    old_part, new_part = payload.split(sep, 1)
                    update_memory(user_id, old_part.strip(), new_part.strip())
                    break
        elif s == "ONBOARDING_COMPLETE":
            set_onboarding_state(user_id, "done")
        else:
            clean_lines.append(line)
    return "\n".join(clean_lines).strip()


# ─── Build messages ───────────────────────────────────────────────────────────

def build_messages(user_id: int, user_text: str, mode: str = "chat") -> list[dict]:
    memories = load_memories(user_id)
    memory_block = ""
    if memories:
        facts = "\n".join(f"  - {f}" for f in memories)
        memory_block = f"\n\nSaved facts about this client:\n{facts}"

    onboarding_note = ""
    if mode == "full":
        onboarding_note = (
            "\n\n⚠️ You are conducting DETAILED ONBOARDING. "
            "STRICT RULE: Send ONE question per message. ONE question mark max. "
            "Do NOT combine questions."
        )
    elif mode == "quick":
        onboarding_note = (
            "\n\n⚠️ You are conducting QUICK ONBOARDING. "
            "STRICT RULE: Send ONE question per message. ONE question mark max. "
            "Do NOT combine questions."
        )

    system  = {"role": "system", "content": SYSTEM_PROMPT + memory_block + onboarding_note}
    history = load_history(user_id)
    new_msg = {"role": "user", "content": user_text}
    return [system] + history + [new_msg]


# ─── OpenAI call ──────────────────────────────────────────────────────────────

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
_locks: dict[int, asyncio.Lock] = {}


def get_lock(user_id: int) -> asyncio.Lock:
    if user_id not in _locks:
        _locks[user_id] = asyncio.Lock()
    return _locks[user_id]


async def ask_ai(user_id: int, user_text: str, mode: str = "chat") -> str:
    messages = build_messages(user_id, user_text, mode)
    kwargs: dict = {"model": MODEL, "input": messages}
    if VECTOR_STORE_ID:
        kwargs["tools"] = [{"type": "file_search", "vector_store_ids": [VECTOR_STORE_ID]}]
    response = await client.responses.create(**kwargs)
    raw      = response.output_text.strip()
    return extract_and_clean(user_id, raw)


# ─── Keyboards ────────────────────────────────────────────────────────────────

def main_menu_keyboard():
    """Inline keyboard shown after onboarding is done."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📋 ჩემი ინფო", callback_data="show_memories"),
            InlineKeyboardButton("🔄 კითხვარი", callback_data="switch_menu"),
        ],
        [
            InlineKeyboardButton("🗑 ინფო წაშლა", callback_data="forget"),
            InlineKeyboardButton("🔃 ისტორია წაშლა", callback_data="reset"),
        ],
    ])


def switch_keyboard():
    """Inline keyboard to switch onboarding mode."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📋 სრული კითხვარი (13)", callback_data="switch_full"),
            InlineKeyboardButton("⚡ სწრაფი კითხვარი (4)", callback_data="switch_quick"),
        ],
        [InlineKeyboardButton("« უკან", callback_data="back_to_menu")],
    ])


def onboarding_keyboard():
    """Keyboard shown at start to choose onboarding mode."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📋 სრული კითხვარი — 13 კითხვა", callback_data="start_full")],
        [InlineKeyboardButton("⚡ სწრაფი კითხვარი — 4 კითხვა",  callback_data="start_quick")],
    ])


# ─── Helpers ──────────────────────────────────────────────────────────────────

async def send_onboarding_choice(update: Update, edit: bool = False):
    text = (
        "👋 გამარჯობა! მე ვარ შენი ფინანსური ასისტენტი.\n\n"
        "როგორ გინდა დავიწყოთ?\n\n"
        "📋 *სრული კითხვარი* — 13 კითხვა, სრული სურათი\n"
        "⚡ *სწრაფი კითხვარი* — 4 კითხვა, სწრაფი დასაწყისი"
    )
    kb = onboarding_keyboard()
    if edit and update.callback_query:
        await update.callback_query.edit_message_text(text, parse_mode="Markdown", reply_markup=kb)
    else:
        await update.effective_message.reply_text(text, parse_mode="Markdown", reply_markup=kb)


async def run_ai_and_reply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    user_text: str,
    mode: str,
    save_user_msg: bool = True,
):
    async with get_lock(user_id):
        await context.bot.send_chat_action(update.effective_chat.id, "typing")
        if save_user_msg:
            save_message(user_id, "user", user_text)
        try:
            reply = await ask_ai(user_id, user_text, mode=mode)
        except Exception as e:
            log.error("AI error [%s]: %s", user_id, e)
            await update.effective_message.reply_text("შეცდომა მოხდა, გთხოვ სცადე თავიდან.")
            return
        save_message(user_id, "assistant", reply)

    ob_state = get_onboarding_state(user_id)
    if ob_state == "done":
        await update.effective_message.reply_text(reply, reply_markup=main_menu_keyboard())
    else:
        await update.effective_message.reply_text(reply)


# ─── Command handlers ─────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id  = update.message.from_user.id
    ob_state = get_onboarding_state(user_id)
    if ob_state == "done" and load_memories(user_id):
        await update.message.reply_text(
            "კვლავ გამარჯობა! 👋 შენი ბიზნეს ინფო შენახული მაქვს.\n\nრით შეგიძლია დაგეხმარო?",
            reply_markup=main_menu_keyboard(),
        )
    else:
        await send_onboarding_choice(update)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📌 *ბრძანებები:*\n\n"
        "/start — მთავარი მენიუ\n"
        "/memories — ჩემი ბიზნეს ინფო\n"
        "/switch — კითხვარის შეცვლა\n"
        "/forget — ინფოს წაშლა\n"
        "/reset — საუბრის ისტორიის წაშლა\n\n"
        "💡 ნებისმიერ დროს შეგიძლია კითხვა დამისვა!",
        parse_mode="Markdown",
    )


async def cmd_memories(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    facts   = load_memories(user_id)
    if not facts:
        await update.message.reply_text("ჯერ არაფერი შენახული მაქვს. დაიწყე /start-ით!")
    else:
        lines = "\n".join(f"• {f}" for f in facts)
        await update.message.reply_text(
            f"📋 *შენი ბიზნეს ინფო:*\n\n{lines}",
            parse_mode="Markdown",
            reply_markup=main_menu_keyboard(),
        )


async def cmd_forget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    clear_memories(user_id)
    set_onboarding_state(user_id, "none")
    await update.message.reply_text("✅ ყველა შენახული ინფო წაიშალა.")
    await send_onboarding_choice(update)


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    clear_history(user_id)
    await update.message.reply_text("✅ საუბრის ისტორია წაიშალა.")


async def cmd_switch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔄 *კითხვარის შეცვლა*\n\n"
        "შენი შენახული ინფო *არ წაიშლება*.\n"
        "მხოლოდ კითხვარის ტიპი იცვლება.",
        parse_mode="Markdown",
        reply_markup=switch_keyboard(),
    )


# ─── Callback query handler ───────────────────────────────────────────────────

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query   = update.callback_query
    user_id = query.from_user.id
    data    = query.data
    await query.answer()

    # ── Onboarding start ──────────────────────────────────────────────────────
    if data == "start_full":
        set_onboarding_state(user_id, "full")
        await query.edit_message_text(
            "📋 *სრული კითხვარი*\n\n"
            "13 კითხვა — ყოველი პასუხი სამუდამოდ შეინახება.\n"
            "თუ კითხვა გაუგებარია, უბრალოდ მკითხე!\n\n"
            "დავიწყოთ 👇",
            parse_mode="Markdown",
        )
        await run_ai_and_reply(
            update, context, user_id,
            "Start the detailed onboarding. Ask ONLY the first question. One question only.",
            mode="full", save_user_msg=False,
        )
        return

    if data == "start_quick":
        set_onboarding_state(user_id, "quick")
        await query.edit_message_text(
            "⚡ *სწრაფი კითხვარი*\n\n"
            "4 კითხვა — სწრაფი დასაწყისი.\n\n"
            "დავიწყოთ 👇",
            parse_mode="Markdown",
        )
        await run_ai_and_reply(
            update, context, user_id,
            "Start the quick onboarding. Ask ONLY the first question. One question only.",
            mode="quick", save_user_msg=False,
        )
        return

    # ── Switch mode (keeps memories) ─────────────────────────────────────────
    if data == "switch_full":
        set_onboarding_state(user_id, "full")
        clear_history(user_id)
        await query.edit_message_text(
            "📋 *სრული კითხვარზე გადავედი*\n\n"
            "შენი ინფო შენახულია ✅\n"
            "ახლა დეტალური კითხვებს გავაგრძელებ.",
            parse_mode="Markdown",
        )
        await run_ai_and_reply(
            update, context, user_id,
            "The user switched to detailed onboarding. Their previous facts are saved. "
            "Continue from where we left off or ask the next missing question. One question only.",
            mode="full", save_user_msg=False,
        )
        return

    if data == "switch_quick":
        set_onboarding_state(user_id, "quick")
        clear_history(user_id)
        await query.edit_message_text(
            "⚡ *სწრაფ კითხვარზე გადავედი*\n\n"
            "შენი ინფო შენახულია ✅\n"
            "ახლა სწრაფ კითხვებს გავაგრძელებ.",
            parse_mode="Markdown",
        )
        await run_ai_and_reply(
            update, context, user_id,
            "The user switched to quick onboarding. Their previous facts are saved. "
            "Continue from where we left off or ask the next missing question. One question only.",
            mode="quick", save_user_msg=False,
        )
        return

    # ── Menu actions ──────────────────────────────────────────────────────────
    if data == "show_memories":
        facts = load_memories(user_id)
        if not facts:
            await query.edit_message_text("ჯერ არაფერი შენახული მაქვს.")
        else:
            lines = "\n".join(f"• {f}" for f in facts)
            await query.edit_message_text(
                f"📋 *შენი ბიზნეს ინფო:*\n\n{lines}",
                parse_mode="Markdown",
                reply_markup=main_menu_keyboard(),
            )
        return

    if data == "switch_menu":
        await query.edit_message_text(
            "🔄 *კითხვარის შეცვლა*\n\n"
            "შენი შენახული ინფო *არ წაიშლება*.\n"
            "მხოლოდ კითხვარის ტიპი იცვლება.",
            parse_mode="Markdown",
            reply_markup=switch_keyboard(),
        )
        return

    if data == "forget":
        clear_memories(user_id)
        set_onboarding_state(user_id, "none")
        await query.edit_message_text("✅ ყველა შენახული ინფო წაიშალა.")
        await send_onboarding_choice(update, edit=False)
        return

    if data == "reset":
        clear_history(user_id)
        await query.edit_message_text(
            "✅ საუბრის ისტორია წაიშალა.",
            reply_markup=main_menu_keyboard(),
        )
        return

    if data == "back_to_menu":
        ob_state = get_onboarding_state(user_id)
        if ob_state == "done":
            await query.edit_message_text(
                "რით შეგიძლია დაგეხმარო?",
                reply_markup=main_menu_keyboard(),
            )
        else:
            await send_onboarding_choice(update, edit=True)
        return


# ─── Message handler ──────────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id   = update.message.from_user.id
    user_text = update.message.text.strip()
    ob_state  = get_onboarding_state(user_id)

    if ob_state == "none":
        await send_onboarding_choice(update)
        return

    mode = ob_state if ob_state in ("full", "quick") else "chat"
    await run_ai_and_reply(update, context, user_id, user_text, mode=mode)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    init_db()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("help",     cmd_help))
    app.add_handler(CommandHandler("memories", cmd_memories))
    app.add_handler(CommandHandler("forget",   cmd_forget))
    app.add_handler(CommandHandler("reset",    cmd_reset))
    app.add_handler(CommandHandler("switch",   cmd_switch))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("Bot is running.")
    app.run_polling()


if __name__ == "__main__":
    main()
