"""
Georgian Financial Assistant Telegram Bot
- SQLite persistent memory (survives Railway restarts)
- Smart onboarding: detailed (13 questions) or quick (4 questions)
- /switch command to change mode without losing data
- Auto-saves business facts from conversation
- Text-based reset with confirmation dialog
- Georgian tax deadline reminders
- /edit command to edit saved facts
- PDF/Image document analysis (max 5MB, 10 pages)
- /export command to export business profile
- Enhanced auto fact extraction
- 50-message chat history
- OpenAI Responses API with File Search (Vector Store)

Environment variables:
    TELEGRAM_TOKEN   — BotFather token
    OPENAI_API_KEY   — OpenAI API key
    VECTOR_STORE_ID  — your vector store ID
"""

import os
import io
import sqlite3
import asyncio
import logging
from datetime import datetime, date
from contextlib import contextmanager

from openai import AsyncOpenAI
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton, Document, PhotoSize
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

MODEL        = "gpt-4.1-mini"
MAX_HISTORY  = 50
DB_PATH      = "/data/memory.db" if os.path.isdir("/data") else "memory.db"
MAX_FILE_MB  = 5
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

RESET_TRIGGERS = {
    "წაშალე", "ყველაფერი წაშალე", "თავიდან დავიწყოთ", "თავიდან",
    "გასუფთავება", "დასუფთავება", "ყველაფერი", "reset", "clear",
    "სრული წაშლა", "ინფო წაშლა", "ახლიდან", "თავიდან ახლიდან",
}

# ─── Georgian tax deadlines ───────────────────────────────────────────────────

def get_upcoming_deadlines(days_ahead: int = 7) -> list[str]:
    today     = date.today()
    year      = today.year
    deadlines = []

    for month in [4, 7, 10]:
        deadlines.append((date(year, month, 15), "მცირე ბიზნესის კვარტალური დეკლარაცია"))
    deadlines.append((date(year + 1, 1, 15), "მცირე ბიზნესის კვარტალური დეკლარაცია"))

    for month in range(1, 13):
        deadlines.append((date(year, month, 15), f"დღგ-ს დეკლარაცია ({month} თვე)"))

    deadlines.append((date(year, 4, 1), "წლიური საშემოსავლო გადასახადის დეკლარაცია"))
    deadlines.append((date(year, 4, 1), "კორპორაციული მოგების გადასახადი"))

    upcoming = []
    for deadline_date, name in deadlines:
        delta = (deadline_date - today).days
        if 0 <= delta <= days_ahead:
            if delta == 0:
                upcoming.append(f"🔴 *დღეს* — {name}")
            elif delta == 1:
                upcoming.append(f"🟠 *ხვალ* — {name}")
            else:
                upcoming.append(f"🟡 *{delta} დღეში* ({deadline_date.strftime('%d.%m')}) — {name}")
    return upcoming


async def send_reminders(app):
    deadlines = get_upcoming_deadlines(days_ahead=7)
    if not deadlines:
        return
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT DISTINCT user_id FROM user_state WHERE onboarding='done'"
        ).fetchall()
    for row in rows:
        user_id = row[0]
        text    = "📅 *საგადასახადო შეხსენება*\n\n" + "\n".join(deadlines)
        try:
            await app.bot.send_message(user_id, text, parse_mode="Markdown")
        except Exception as e:
            log.warning("Could not send reminder to %s: %s", user_id, e)


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
When the user shares business facts (in ANY message, not just onboarding),
extract and save them by adding at the END of your response:

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
SAVE: founded — 2022
SAVE: industry — software development

Always extract facts even from casual conversation. Examples:
- "ჩვენ 5 თანამშრომელი გვყავს" → SAVE: employees — 5
- "მე ვარ დირექტორი" → SAVE: owner/director — yes
- "ბრუნვა გაიზარდა 20,000 ლარამდე" → SAVE: monthly revenue — 20,000 GEL

If the user CORRECTS a previously saved fact:
UPDATE: [old fact keyword] → [new fact]

## ⚠️ CRITICAL ONBOARDING RULES
RULE 1: Send EXACTLY ONE question per message. ONE. Never two. Never three.
RULE 2: Do NOT combine questions with "და", "ასევე", "გარდა ამისა", "and", "also".
RULE 3: Count the "?" marks in your message. If there are 2 or more — rewrite.
RULE 4: After the user answers, write SAVE: for that fact, then ask the NEXT single question.
RULE 5: If user asks for clarification, answer it, then ask the SAME question again.
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

DOCUMENT_PROMPT = """You are analyzing a document uploaded by a Georgian business owner.

Extract ALL relevant business/financial information from this document and:
1. Summarize what the document contains in 2-3 sentences
2. List key financial figures, dates, and facts found
3. Answer the user's question about the document if they asked one
4. Save any important business facts using SAVE: format

Always respond in the same language the user writes in.
Be specific — quote exact numbers, dates, and names from the document.

At the end, save relevant facts:
SAVE: [fact from document]
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


def load_memories_with_keys(user_id: int) -> list[tuple[str, str]]:
    with get_db() as db:
        rows = db.execute(
            "SELECT mem_key, fact FROM memories WHERE user_id = ? ORDER BY created_at",
            (user_id,)
        ).fetchall()
    return [(row["mem_key"], row["fact"]) for row in rows]


def save_memory(user_id: int, fact: str):
    fact    = fact.strip()
    mem_key = fact.split(" — ")[0].strip().lower() if " — " in fact else fact[:40].lower()
    with get_db() as db:
        db.execute("DELETE FROM memories WHERE user_id = ? AND mem_key = ?", (user_id, mem_key))
        db.execute("INSERT INTO memories(user_id, mem_key, fact) VALUES(?, ?, ?)", (user_id, mem_key, fact))
    log.info("Memory saved [%s]: %s", user_id, fact)


def delete_memory_by_key(user_id: int, mem_key: str):
    with get_db() as db:
        db.execute("DELETE FROM memories WHERE user_id = ? AND mem_key = ?", (user_id, mem_key))


def update_memory(user_id: int, old_keyword: str, new_fact: str):
    new_fact = new_fact.strip()
    old_key  = old_keyword.strip().lower()
    new_key  = new_fact.split(" — ")[0].strip().lower() if " — " in new_fact else old_key
    with get_db() as db:
        db.execute("DELETE FROM memories WHERE user_id = ? AND mem_key = ?", (user_id, old_key))
        db.execute("INSERT INTO memories(user_id, mem_key, fact) VALUES(?, ?, ?)", (user_id, new_key, new_fact))


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
            "\n\n⚠️ DETAILED ONBOARDING MODE. "
            "STRICT: ONE question per message. ONE '?' max. No combining."
        )
    elif mode == "quick":
        onboarding_note = (
            "\n\n⚠️ QUICK ONBOARDING MODE. "
            "STRICT: ONE question per message. ONE '?' max. No combining."
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


async def analyze_document(user_id: int, file_bytes: bytes, mime_type: str, caption: str = "") -> str:
    """Analyze uploaded document using OpenAI."""
    import base64

    user_question = caption if caption else "გაანალიზე ეს დოკუმენტი და მითხარი რა შეიცავს."

    memories = load_memories(user_id)
    memory_block = ""
    if memories:
        facts = "\n".join(f"  - {f}" for f in memories)
        memory_block = f"\n\nClient's saved facts:\n{facts}"

    system_content = DOCUMENT_PROMPT + memory_block

    # Build content based on file type
    if mime_type in ("image/jpeg", "image/png", "image/webp"):
        b64 = base64.b64encode(file_bytes).decode()
        user_content = [
            {
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{b64}",
            },
            {"type": "input_text", "text": user_question},
        ]
    else:
        # PDF or other document — send as file
        b64 = base64.b64encode(file_bytes).decode()
        user_content = [
            {
                "type": "input_file",
                "filename": "document.pdf",
                "file_data": f"data:{mime_type};base64,{b64}",
            },
            {"type": "input_text", "text": user_question},
        ]

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]

    response = await client.responses.create(model=MODEL, input=messages)
    raw      = response.output_text.strip()
    return extract_and_clean(user_id, raw)


# ─── Export helper ────────────────────────────────────────────────────────────

def build_export_text(user_id: int) -> str:
    facts = load_memories(user_id)
    if not facts:
        return ""
    lines = "\n".join(f"• {f}" for f in facts)
    now   = datetime.now().strftime("%d.%m.%Y %H:%M")
    return (
        f"📋 *ბიზნეს პროფილი*\n"
        f"_გენერირებულია: {now}_\n\n"
        f"{lines}\n\n"
        f"_ეს ინფო შენახულია შენი ბოტის მეხსიერებაში._"
    )


# ─── Keyboards ────────────────────────────────────────────────────────────────

def main_menu_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📋 ჩემი ინფო",   callback_data="show_memories"),
            InlineKeyboardButton("✏️ რედაქტირება",  callback_data="edit_memories"),
        ],
        [
            InlineKeyboardButton("🔄 კითხვარი",     callback_data="switch_menu"),
            InlineKeyboardButton("📅 ვადები",        callback_data="show_deadlines"),
        ],
        [
            InlineKeyboardButton("📤 ექსპორტი",     callback_data="export_profile"),
            InlineKeyboardButton("🗑 ინფო წაშლა",   callback_data="forget"),
        ],
    ])


def confirm_reset_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ კი, წაშალე",      callback_data="confirm_reset"),
            InlineKeyboardButton("❌ არა, გავაგრძელო", callback_data="cancel_reset"),
        ]
    ])


def switch_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📋 სრული კითხვარი (13)", callback_data="switch_full")],
        [InlineKeyboardButton("⚡ სწრაფი კითხვარი (4)",  callback_data="switch_quick")],
        [InlineKeyboardButton("« უკან",                  callback_data="back_to_menu")],
    ])


def onboarding_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📋 სრული კითხვარი — 13 კითხვა", callback_data="start_full")],
        [InlineKeyboardButton("⚡ სწრაფი კითხვარი — 4 კითხვა",  callback_data="start_quick")],
    ])


def edit_list_keyboard(memories: list[tuple[str, str]]):
    buttons = []
    for mem_key, fact in memories:
        label = fact[:38] + ("…" if len(fact) > 38 else "")
        buttons.append([InlineKeyboardButton(f"🗑 {label}", callback_data=f"del_fact:{mem_key}")])
    buttons.append([InlineKeyboardButton("« უკან", callback_data="back_to_menu")])
    return InlineKeyboardMarkup(buttons)


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


async def do_full_reset(user_id: int, update: Update):
    clear_memories(user_id)
    clear_history(user_id)
    set_onboarding_state(user_id, "none")
    await update.effective_message.reply_text("✅ ყველა ინფო და ისტორია წაიშალა!")
    await send_onboarding_choice(update)


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
        "/edit — ინფოს რედაქტირება\n"
        "/export — ბიზნეს პროფილის ექსპორტი\n"
        "/switch — კითხვარის შეცვლა\n"
        "/deadlines — საგადასახადო ვადები\n"
        "/forget — ინფოს წაშლა\n"
        "/reset — საუბრის ისტორიის წაშლა\n\n"
        "📎 *დოკუმენტები:*\n"
        "გამომიგზავნე PDF, Word ან ფოტო (მაქს. 5MB)\n"
        "დავანალიზებ და ინფოს ამოვიღებ!\n\n"
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


async def cmd_edit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id  = update.message.from_user.id
    memories = load_memories_with_keys(user_id)
    if not memories:
        await update.message.reply_text("ჯერ არაფერი შენახული მაქვს.")
        return
    await update.message.reply_text(
        "✏️ *რედაქტირება*\n\nდააჭირე ფაქტს წასაშლელად:",
        parse_mode="Markdown",
        reply_markup=edit_list_keyboard(memories),
    )


async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    text    = build_export_text(user_id)
    if not text:
        await update.message.reply_text("ჯერ არაფერი შენახული მაქვს.")
        return
    await update.message.reply_text(text, parse_mode="Markdown", reply_markup=main_menu_keyboard())


async def cmd_forget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "⚠️ *დარწმუნებული ხარ?*\n\nყველა შენახული ინფო და საუბრის ისტორია წაიშლება.",
        parse_mode="Markdown",
        reply_markup=confirm_reset_keyboard(),
    )


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    clear_history(user_id)
    await update.message.reply_text("✅ საუბრის ისტორია წაიშალა.")


async def cmd_switch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔄 *კითხვარის შეცვლა*\n\nშენი შენახული ინფო *არ წაიშლება*.",
        parse_mode="Markdown",
        reply_markup=switch_keyboard(),
    )


async def cmd_deadlines(update: Update, context: ContextTypes.DEFAULT_TYPE):
    deadlines = get_upcoming_deadlines(days_ahead=30)
    if not deadlines:
        await update.message.reply_text("✅ მომავალ 30 დღეში საგადასახადო ვადები არ არის.")
    else:
        text = "📅 *მომავალი საგადასახადო ვადები:*\n\n" + "\n".join(deadlines)
        await update.message.reply_text(text, parse_mode="Markdown")


# ─── Document handler ─────────────────────────────────────────────────────────

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    caption = update.message.caption or ""

    ob_state = get_onboarding_state(user_id)
    if ob_state == "none":
        await send_onboarding_choice(update)
        return

    # Determine file type and get file object
    if update.message.document:
        doc       = update.message.document
        mime_type = doc.mime_type or "application/octet-stream"
        file_size = doc.file_size or 0
        file_obj  = doc

        allowed_mimes = (
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "image/jpeg", "image/png", "image/webp",
        )
        if mime_type not in allowed_mimes:
            await update.message.reply_text(
                "❌ ეს ფაილის ტიპი არ არის მხარდაჭერილი.\n\n"
                "გამომიგზავნე: PDF, Word (.docx), ან ფოტო (JPG/PNG)"
            )
            return

    elif update.message.photo:
        photo     = update.message.photo[-1]  # highest resolution
        file_size = photo.file_size or 0
        mime_type = "image/jpeg"
        file_obj  = photo
    else:
        return

    # Check file size
    if file_size > MAX_FILE_BYTES:
        await update.message.reply_text(
            f"❌ ფაილი ძალიან დიდია ({file_size // (1024*1024):.1f}MB).\n"
            f"მაქსიმუმი: {MAX_FILE_MB}MB"
        )
        return

    await context.bot.send_chat_action(update.effective_chat.id, "typing")
    await update.message.reply_text("📄 ვკითხულობ დოკუმენტს...")

    try:
        tg_file    = await context.bot.get_file(file_obj.file_id)
        file_bytes = await tg_file.download_as_bytearray()
        file_bytes = bytes(file_bytes)
    except Exception as e:
        log.error("File download error [%s]: %s", user_id, e)
        await update.message.reply_text("❌ ფაილის ჩამოტვირთვა ვერ მოხერხდა.")
        return

    try:
        reply = await analyze_document(user_id, file_bytes, mime_type, caption)
    except Exception as e:
        log.error("Document analysis error [%s]: %s", user_id, e)
        await update.message.reply_text("❌ დოკუმენტის ანალიზი ვერ მოხერხდა. სცადე თავიდან.")
        return

    save_message(user_id, "user", f"[დოკუმენტი] {caption}")
    save_message(user_id, "assistant", reply)

    ob_state = get_onboarding_state(user_id)
    if ob_state == "done":
        await update.message.reply_text(reply, reply_markup=main_menu_keyboard())
    else:
        await update.message.reply_text(reply)


# ─── Callback query handler ───────────────────────────────────────────────────

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query   = update.callback_query
    user_id = query.from_user.id
    data    = query.data
    await query.answer()

    if data == "confirm_reset":
        await query.edit_message_text("⏳ იშლება...")
        await do_full_reset(user_id, update)
        return

    if data == "cancel_reset":
        await query.edit_message_text("❌ გაუქმდა. ყველაფერი ისევ ადგილზეა ✅")
        return

    if data.startswith("del_fact:"):
        mem_key = data[9:]
        delete_memory_by_key(user_id, mem_key)
        memories = load_memories_with_keys(user_id)
        if not memories:
            await query.edit_message_text("✅ ყველა ფაქტი წაიშალა.")
        else:
            lines = "\n".join(f"• {f}" for _, f in memories)
            await query.edit_message_text(
                f"✅ წაიშალა.\n\n📋 *დარჩენილი ინფო:*\n\n{lines}",
                parse_mode="Markdown",
                reply_markup=edit_list_keyboard(memories),
            )
        return

    if data == "start_full":
        set_onboarding_state(user_id, "full")
        await query.edit_message_text(
            "📋 *სრული კითხვარი*\n\n13 კითხვა — ყოველი პასუხი სამუდამოდ შეინახება.\n\nდავიწყოთ 👇",
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
            "⚡ *სწრაფი კითხვარი*\n\n4 კითხვა — სწრაფი დასაწყისი.\n\nდავიწყოთ 👇",
            parse_mode="Markdown",
        )
        await run_ai_and_reply(
            update, context, user_id,
            "Start the quick onboarding. Ask ONLY the first question. One question only.",
            mode="quick", save_user_msg=False,
        )
        return

    if data == "switch_full":
        set_onboarding_state(user_id, "full")
        clear_history(user_id)
        await query.edit_message_text(
            "📋 *სრულ კითხვარზე გადავედი*\n\nშენი ინფო შენახულია ✅",
            parse_mode="Markdown",
        )
        await run_ai_and_reply(
            update, context, user_id,
            "User switched to detailed onboarding. Previous facts are saved. "
            "Ask the next missing question only. ONE question.",
            mode="full", save_user_msg=False,
        )
        return

    if data == "switch_quick":
        set_onboarding_state(user_id, "quick")
        clear_history(user_id)
        await query.edit_message_text(
            "⚡ *სწრაფ კითხვარზე გადავედი*\n\nშენი ინფო შენახულია ✅",
            parse_mode="Markdown",
        )
        await run_ai_and_reply(
            update, context, user_id,
            "User switched to quick onboarding. Previous facts are saved. "
            "Ask the next missing question only. ONE question.",
            mode="quick", save_user_msg=False,
        )
        return

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

    if data == "edit_memories":
        memories = load_memories_with_keys(user_id)
        if not memories:
            await query.edit_message_text("ჯერ არაფერი შენახული მაქვს.")
        else:
            await query.edit_message_text(
                "✏️ *რედაქტირება*\n\nდააჭირე ფაქტს წასაშლელად:",
                parse_mode="Markdown",
                reply_markup=edit_list_keyboard(memories),
            )
        return

    if data == "export_profile":
        text = build_export_text(user_id)
        if not text:
            await query.edit_message_text("ჯერ არაფერი შენახული მაქვს.")
        else:
            await query.edit_message_text(
                text, parse_mode="Markdown", reply_markup=main_menu_keyboard()
            )
        return

    if data == "show_deadlines":
        deadlines = get_upcoming_deadlines(days_ahead=30)
        text = (
            "📅 *მომავალი საგადასახადო ვადები:*\n\n" + "\n".join(deadlines)
            if deadlines else "✅ მომავალ 30 დღეში საგადასახადო ვადები არ არის."
        )
        await query.edit_message_text(
            text, parse_mode="Markdown", reply_markup=main_menu_keyboard()
        )
        return

    if data == "switch_menu":
        await query.edit_message_text(
            "🔄 *კითხვარის შეცვლა*\n\nშენი შენახული ინფო *არ წაიშლება*.",
            parse_mode="Markdown",
            reply_markup=switch_keyboard(),
        )
        return

    if data == "forget":
        await query.edit_message_text(
            "⚠️ *დარწმუნებული ხარ?*\n\nყველა შენახული ინფო და საუბრის ისტორია წაიშლება.",
            parse_mode="Markdown",
            reply_markup=confirm_reset_keyboard(),
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

    if user_text.lower() in {t.lower() for t in RESET_TRIGGERS}:
        await update.message.reply_text(
            "⚠️ *დარწმუნებული ხარ?*\n\nყველა შენახული ინფო და საუბრის ისტორია წაიშლება.",
            parse_mode="Markdown",
            reply_markup=confirm_reset_keyboard(),
        )
        return

    ob_state = get_onboarding_state(user_id)
    if ob_state == "none":
        await send_onboarding_choice(update)
        return

    mode = ob_state if ob_state in ("full", "quick") else "chat"
    await run_ai_and_reply(update, context, user_id, user_text, mode=mode)


# ─── Daily reminder job ───────────────────────────────────────────────────────

async def daily_reminder_job(context: ContextTypes.DEFAULT_TYPE):
    await send_reminders(context.application)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    init_db()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",     cmd_start))
    app.add_handler(CommandHandler("help",      cmd_help))
    app.add_handler(CommandHandler("memories",  cmd_memories))
    app.add_handler(CommandHandler("edit",      cmd_edit))
    app.add_handler(CommandHandler("export",    cmd_export))
    app.add_handler(CommandHandler("forget",    cmd_forget))
    app.add_handler(CommandHandler("reset",     cmd_reset))
    app.add_handler(CommandHandler("switch",    cmd_switch))
    app.add_handler(CommandHandler("deadlines", cmd_deadlines))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.PHOTO, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.job_queue.run_daily(
        daily_reminder_job,
        time=datetime.strptime("09:00", "%H:%M").time(),
        name="daily_reminders",
    )

    log.info("Bot is running.")
    app.run_polling()


if __name__ == "__main__":
    main()
