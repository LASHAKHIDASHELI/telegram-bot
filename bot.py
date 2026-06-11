"""
Georgian Financial Assistant Telegram Bot
- SQLite persistent memory (survives restarts)
- Smart onboarding: detailed (13 questions) or quick (4 questions)
- Inline clarification: user can ask questions mid-onboarding
- 50-message chat history
- OpenAI File Search (Vector Store) for accounting docs
- Auto-updates facts when user corrects info

Environment variables:
    TELEGRAM_TOKEN   — BotFather token
    OPENAI_API_KEY   — OpenAI API key
    VECTOR_STORE_ID  — your vector store ID (optional)
"""

import os
import re
import sqlite3
import asyncio
import logging
from contextlib import contextmanager

from openai import AsyncOpenAI
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# ─── Config ──────────────────────────────────────────────────────────────────

TELEGRAM_TOKEN  = os.environ["TELEGRAM_TOKEN"]
OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
VECTOR_STORE_ID = os.environ.get("VECTOR_STORE_ID", "")

MODEL       = "gpt-4.1-mini"
MAX_HISTORY = 50   # messages kept in DB per user
DB_PATH     = "/data/memory.db" if os.path.isdir("/data") else "memory.db"

# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Georgian accounting and financial AI assistant 
with 20+ years of experience. You help startups with:
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

## MEMORY MANAGEMENT
When the user shares business facts, extract and save them with this format at the 
END of your response (hidden from display but parsed by the system):

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

If the user CORRECTS or UPDATES a fact:
UPDATE: [old fact keyword] → [new fact]

Example:
UPDATE: monthly revenue → monthly revenue — 25,000 GEL

## ONBOARDING
When a user is in onboarding mode, you are conducting a professional intake interview.
Ask questions ONE AT A TIME. Never combine questions.
Wait for each answer. Adapt based on what you learn.
If an answer is vague, ask a clarifying follow-up before moving on.
After EACH answer, save the relevant facts using SAVE: format.

If the user asks a clarification question mid-onboarding, answer it fully and 
professionally, then continue the onboarding from where you left off.

When onboarding is complete (all necessary info gathered), end with:
ONBOARDING_COMPLETE

## DETAILED ONBOARDING (13 questions)
Ask exactly these questions, one at a time, in this order:

Q1:  "რა ჰქვია შენს კომპანიას და რა სფეროში მუშაობს?"
Q2:  "სამართლებრივი ფორმა რა არის და საგადასახადო სამსახურში რეგისტრირებული ხარ?"
Q3:  "რომელ საგადასახადო რეჟიმზე ხარ — მცირე ბიზნესი 1%, სტანდარტული 15%, ვირტუალური ზონა, თუ სხვა?"
Q4:  "დღგ-ს გადამხდელი ხარ და წლიური ბრუნვა 100,000 ლარს გადააჭარბებს?"
Q5:  "ყოველთვიური საშუალო შემოსავალი რამდენია და რა ვალუტაში?"
Q6:  "შემოსავალი ერთი წყაროდან მოდის თუ რამდენიმედან — მომსახურება, პროდუქტი, სააბონენტო?"
Q7:  "კლიენტები ქართული კომპანიები არიან, უცხოური, თუ ორივე?"
Q8:  "თანამშრომლები გყავს და ოფიციალური ხელფასი ეძლევათ?"
Q9:  "ძირითადი ხარჯების კატეგორიები რა არის — ხელფასი, ქირა, მარკეტინგი, ტექნოლოგია?"
Q10: "ბუღალტერი გყავს თუ თავად აწარმოებ აღრიცხვას?"
Q11: "ბოლო საგადასახადო დეკლარაცია წარდგენილი გაქვს და რა პერიოდზე?"
Q12: "საბანკო სესხი, ინვესტიცია ან გრანტი გაქვს?"
Q13: "ამჟამად ყველაზე მწვავე ფინანსური პრობლემა ან კითხვა რა არის?"

After each answer save the relevant facts with SAVE: format.
If the answer is vague, ask one follow-up before moving to the next question.
If the user asks a clarification question at any point, answer it fully, then continue from where you left off.
After Q13 end with ONBOARDING_COMPLETE.

## QUICK ONBOARDING (4 questions)
Ask exactly these questions, one at a time, in this order:

Q1: "ბიზნესი რეგისტრირებულია და რა ფორმით — შპს, ინდ.მეწარმე, არარეგისტრირებული?"
Q2: "ყოველთვიური ბრუნვა დაახლოებით რამდენია?"
Q3: "თანამშრომლები გყავს?"
Q4: "რა გჭირდება ახლა — გადასახადები, დეკლარაცია, ფინანსური გეგმა, თუ სხვა?"

After each answer save the relevant facts with SAVE: format.
After Q4, if their situation seems complex, offer the full detailed interview.
Then end with ONBOARDING_COMPLETE.
"""

ONBOARDING_INTRO_FULL = """გამარჯობა! მე ვარ შენი პირადი ფინანსური და ბუღალტრული ასისტენტი.

რომ შენს ბიზნესს ზუსტი და პერსონალური დახმარება გავუწიო, ჯერ რამდენიმე კითხვა მინდა დავუსვა.

პასუხები სამუდამოდ შეინახება — მომავალში ყოველ ჯერზე ამ ინფოს გამოვიყენებ.

თუ რომელიმე კითხვა გაუგებარია, იქვე დამისვი — ავხსნი და გავაგრძელებ.

დავიწყოთ! 👇"""

ONBOARDING_INTRO_QUICK = """გამარჯობა! მე ვარ შენი ფინანსური ასისტენტი.

სწრაფად — 4 კითხვა და მაშინვე შეგვიძლია საქმეზე გადავიდეთ. 👇"""

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

            CREATE TABLE IF NOT EXISTS memories (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                key        TEXT    NOT NULL,
                fact       TEXT    NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, key) ON CONFLICT REPLACE
            );

            CREATE TABLE IF NOT EXISTS user_state (
                user_id    INTEGER PRIMARY KEY,
                onboarding TEXT    DEFAULT 'none',
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_conv_user   ON conversations(user_id);
            CREATE INDEX IF NOT EXISTS idx_memory_user ON memories(user_id);
        """)
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
    """Returns: 'none', 'full', 'quick', 'done'"""
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
    """Parse 'key — value' format and upsert."""
    fact = fact.strip()
    if " — " in fact:
        key = fact.split(" — ")[0].strip().lower()
    else:
        key = fact[:40].lower()

    with get_db() as db:
        db.execute(
            """INSERT INTO memories(user_id, key, fact)
               VALUES(?, ?, ?)
               ON CONFLICT(user_id, key) DO UPDATE SET
                   fact=excluded.fact,
                   created_at=CURRENT_TIMESTAMP""",
            (user_id, key, fact)
        )
    log.info("Memory saved [%s]: %s", user_id, fact)


def update_memory(user_id: int, old_keyword: str, new_fact: str):
    """Find memory by keyword and replace it."""
    new_fact = new_fact.strip()
    old_key  = old_keyword.strip().lower()

    if " — " in new_fact:
        new_key = new_fact.split(" — ")[0].strip().lower()
    else:
        new_key = old_key

    with get_db() as db:
        # delete old
        db.execute(
            "DELETE FROM memories WHERE user_id = ? AND key = ?",
            (user_id, old_key)
        )
        # insert new
        db.execute(
            "INSERT INTO memories(user_id, key, fact) VALUES(?, ?, ?)",
            (user_id, new_key, new_fact)
        )
    log.info("Memory updated [%s]: %s → %s", user_id, old_keyword, new_fact)


def clear_memories(user_id: int):
    with get_db() as db:
        db.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))


# ─── Conversation helpers ─────────────────────────────────────────────────────

def load_history(user_id: int) -> list[dict]:
    with get_db() as db:
        rows = db.execute(
            """SELECT role, content FROM conversations
               WHERE user_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (user_id, MAX_HISTORY)
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def save_message(user_id: int, role: str, content: str):
    with get_db() as db:
        db.execute(
            "INSERT INTO conversations(user_id, role, content) VALUES(?, ?, ?)",
            (user_id, role, content)
        )
        # keep only last MAX_HISTORY messages
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


# ─── Parse AI response for SAVE/UPDATE directives ────────────────────────────

def extract_and_clean(user_id: int, raw_text: str) -> str:
    """
    Pull SAVE: and UPDATE: lines from AI response,
    process them into DB, and return clean text for display.
    """
    lines       = raw_text.split("\n")
    clean_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("SAVE:"):
            fact = stripped[5:].strip()
            if fact:
                save_memory(user_id, fact)

        elif stripped.startswith("UPDATE:"):
            payload = stripped[7:].strip()
            if " → " in payload:
                old_part, new_part = payload.split(" → ", 1)
                update_memory(user_id, old_part.strip(), new_part.strip())

        elif stripped == "ONBOARDING_COMPLETE":
            set_onboarding_state(user_id, "done")

        else:
            clean_lines.append(line)

    return "\n".join(clean_lines).strip()


# ─── Build messages for OpenAI ────────────────────────────────────────────────

def build_messages(user_id: int, user_text: str, mode: str = "chat") -> list[dict]:
    memories = load_memories(user_id)
    memory_block = ""
    if memories:
        facts = "\n".join(f"  - {f}" for f in memories)
        memory_block = f"\n\nSaved facts about this client:\n{facts}"

    onboarding_instruction = ""
    if mode == "full":
        onboarding_instruction = (
            "\n\nYou are currently conducting the DETAILED ONBOARDING (13 questions). "
            "Ask questions one at a time. Save facts after each answer. "
            "If user asks a clarification question, answer it then continue onboarding."
        )
    elif mode == "quick":
        onboarding_instruction = (
            "\n\nYou are currently conducting the QUICK ONBOARDING (4 questions). "
            "Ask questions one at a time. Save facts after each answer."
        )

    system = {
        "role": "system",
        "content": SYSTEM_PROMPT + memory_block + onboarding_instruction
    }

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

    kwargs = dict(model=MODEL, input=messages)

    if VECTOR_STORE_ID:
        kwargs["tools"] = [{
            "type": "file_search",
            "vector_store_ids": [VECTOR_STORE_ID]
        }]

    response = await client.responses.create(**kwargs)
    raw = response.output_text.strip()

    clean = extract_and_clean(user_id, raw)
    return clean


# ─── Telegram handlers ────────────────────────────────────────────────────────

MODE_KEYBOARD = ReplyKeyboardMarkup(
    [["📋 სრული კითხვარი", "⚡ სწრაფი კითხვარი"]],
    resize_keyboard=True,
    one_time_keyboard=True
)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    ob_state = get_onboarding_state(user_id)

    if ob_state == "done":
        memories = load_memories(user_id)
        if memories:
            await update.message.reply_text(
                "კვლავ გამარჯობა! შენი ბიზნეს ინფო შენახული მაქვს. "
                "რით შეგიძლია დაგეხმარო?",
                reply_markup=ReplyKeyboardRemove()
            )
        else:
            await _offer_onboarding(update)
    else:
        await _offer_onboarding(update)


async def _offer_onboarding(update: Update):
    await update.message.reply_text(
        "გამარჯობა! 👋 მე ვარ შენი ფინანსური ასისტენტი.\n\n"
        "როგორ გინდა დავიწყოთ?\n\n"
        "📋 *სრული კითხვარი* — 13 კითხვა, სრული სურათი\n"
        "⚡ *სწრაფი კითხვარი* — 4 კითხვა, სწრაფი დასაწყისი",
        parse_mode="Markdown",
        reply_markup=MODE_KEYBOARD
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📌 *ბრძანებები:*\n\n"
        "/start — თავიდან დაწყება\n"
        "/memories — რას ვიცი შენი ბიზნესის შესახებ\n"
        "/forget — შენახული ინფოს წაშლა\n"
        "/reset — საუბრის ისტორიის წაშლა\n"
        "/restart — კითხვარის თავიდან დაწყება\n\n"
        "💡 ნებისმიერ დროს შეგიძლია კითხვა დამისვა!",
        parse_mode="Markdown"
    )


async def cmd_memories(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    facts   = load_memories(user_id)
    if not facts:
        await update.message.reply_text(
            "შენი ბიზნესის შესახებ ჯერ არაფერი შენახული მაქვს.\n"
            "დაიწყე /start-ით!"
        )
    else:
        lines = "\n".join(f"• {f}" for f in facts)
        await update.message.reply_text(
            f"📋 *რას ვიცი შენი ბიზნესის შესახებ:*\n\n{lines}",
            parse_mode="Markdown"
        )


async def cmd_forget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    clear_memories(user_id)
    set_onboarding_state(user_id, "none")
    await update.message.reply_text("✅ ყველა შენახული ინფო წაიშალა.")


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    clear_history(user_id)
    await update.message.reply_text("✅ საუბრის ისტორია წაიშალა.")


async def cmd_restart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    set_onboarding_state(user_id, "none")
    await _offer_onboarding(update)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id   = update.message.from_user.id
    user_text = update.message.text.strip()

    ob_state = get_onboarding_state(user_id)

    # ── Mode selection buttons ────────────────────────────────────────────────
    if user_text in ("📋 სრული კითხვარი", "სრული კითხვარი", "სრული", "detailed"):
        set_onboarding_state(user_id, "full")
        await update.message.reply_text(
            ONBOARDING_INTRO_FULL,
            reply_markup=ReplyKeyboardRemove()
        )
        async with get_lock(user_id):
            await context.bot.send_chat_action(update.effective_chat.id, "typing")
            try:
                reply = await ask_ai(
                    user_id,
                    "Start the detailed onboarding. Ask the first question.",
                    mode="full"
                )
            except Exception as e:
                log.error("AI error [%s]: %s", user_id, e)
                await update.message.reply_text("შეცდომა, სცადე თავიდან.")
                return
            save_message(user_id, "assistant", reply)
        await update.message.reply_text(reply)
        return

    if user_text in ("⚡ სწრაფი კითხვარი", "სწრაფი კითხვარი", "სწრაფი", "quick"):
        set_onboarding_state(user_id, "quick")
        await update.message.reply_text(
            ONBOARDING_INTRO_QUICK,
            reply_markup=ReplyKeyboardRemove()
        )
        async with get_lock(user_id):
            await context.bot.send_chat_action(update.effective_chat.id, "typing")
            try:
                reply = await ask_ai(
                    user_id,
                    "Start the quick onboarding. Ask the first question.",
                    mode="quick"
                )
            except Exception as e:
                log.error("AI error [%s]: %s", user_id, e)
                await update.message.reply_text("შეცდომა, სცადე თავიდან.")
                return
            save_message(user_id, "assistant", reply)
        await update.message.reply_text(reply)
        return

    # ── No mode chosen yet — prompt ───────────────────────────────────────────
    if ob_state == "none":
        await _offer_onboarding(update)
        return

    # ── Normal message (onboarding in progress or done) ───────────────────────
    mode = ob_state if ob_state in ("full", "quick") else "chat"

    async with get_lock(user_id):
        await context.bot.send_chat_action(update.effective_chat.id, "typing")
        save_message(user_id, "user", user_text)

        try:
            reply = await ask_ai(user_id, user_text, mode=mode)
        except Exception as e:
            log.error("AI error [%s]: %s", user_id, e)
            await update.message.reply_text("შეცდომა, სცადე თავიდან.")
            return

        save_message(user_id, "assistant", reply)

    await update.message.reply_text(reply)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    init_db()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("help",     cmd_help))
    app.add_handler(CommandHandler("memories", cmd_memories))
    app.add_handler(CommandHandler("forget",   cmd_forget))
    app.add_handler(CommandHandler("reset",    cmd_reset))
    app.add_handler(CommandHandler("restart",  cmd_restart))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("Bot is running.")
    app.run_polling()


if __name__ == "__main__":
    main()
