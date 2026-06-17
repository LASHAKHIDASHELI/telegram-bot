"""
Georgian Financial Assistant Telegram Bot
- SQLite persistent memory (survives Railway restarts)
- Smart onboarding: detailed (13 questions) or quick (4 questions)
- /switch command to change mode without losing data
- Auto-saves business facts from conversation
- Text-based reset with confirmation dialog
- Georgian tax deadline reminders (personalized by tax regime)
- /edit command to edit saved facts
- Monthly financial tracker (/tracker)
- Multi-language support (Georgian/English/Russian)
- Enhanced auto fact extraction
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
from datetime import datetime, date
from contextlib import contextmanager

from openai import AsyncOpenAI
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
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

MODEL          = "gpt-4.1-mini"
MAX_HISTORY    = 50
DB_PATH        = "/data/memory.db" if os.path.isdir("/data") else "memory.db"

RESET_TRIGGERS = {
    "წაშალე", "ყველაფერი წაშალე", "თავიდან დავიწყოთ", "თავიდან",
    "გასუფთავება", "დასუფთავება", "ყველაფერი", "reset", "clear",
    "სრული წაშლა", "ინფო წაშლა", "ახლიდან", "თავიდან ახლიდან",
}

# ─── Georgian tax deadlines (personalized) ────────────────────────────────────

def get_upcoming_deadlines(days_ahead: int = 7, tax_regime: str = "") -> list[str]:
    today     = date.today()
    year      = today.year
    deadlines = []
    regime    = tax_regime.lower()

    # Small business — monthly by 15th
    if "small business" in regime or "1%" in regime or "მცირე" in regime or not regime:
        for month in range(1, 13):
            deadlines.append((
                date(year, month, 15),
                "მცირე ბიზნესის ყოველთვიური დეკლარაცია (rs.ge)"
            ))

    # VAT — monthly by 15th
    if "vat" in regime or "დღგ" in regime or not regime:
        for month in range(1, 13):
            deadlines.append((
                date(year, month, 15),
                f"დღგ-ს დეკლარაცია — Form 300 ({month} თვე)"
            ))

    # Annual income declaration — April 1
    deadlines.append((date(year, 4, 1), "წლიური საშემოსავლო დეკლარაცია (1 აპრილი)"))

    # SARAS — October 1
    deadlines.append((date(year, 10, 1), "SARAS ფინანსური ანგარიშგება (1 ოქტომბერი)"))

    # Pension — monthly by 15th
    for month in range(1, 13):
        deadlines.append((
            date(year, month, 15),
            f"საპენსიო/ხელფასის დეკლარაცია — Form 200 ({month} თვე)"
        ))

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

    # Remove duplicates
    seen = set()
    unique = []
    for item in upcoming:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


async def send_reminders(app):
    """Send personalized reminders based on user's tax regime."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT DISTINCT user_id FROM user_state WHERE onboarding='done'"
        ).fetchall()

    for row in rows:
        user_id = row[0]
        # Get user's tax regime from memories
        with sqlite3.connect(DB_PATH) as conn:
            mem_row = conn.execute(
                "SELECT fact FROM memories WHERE user_id=? AND mem_key LIKE '%tax regime%'",
                (user_id,)
            ).fetchone()
        regime    = mem_row[0] if mem_row else ""
        deadlines = get_upcoming_deadlines(days_ahead=7, tax_regime=regime)
        if not deadlines:
            continue
        text = "📅 *საგადასახადო შეხსენება*\n\n" + "\n".join(deadlines)
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
- Step-by-step guidance on rs.ge portal navigation

You have access to uploaded accounting and financial documents via file search.
You also have access to saved facts about this specific user's business.
Use both sources to give accurate, personalized advice.

## ⚠️ CRITICAL RULE — RECENT LAW CHANGES (2025-2026)
Your training knowledge is reliable only up to mid-2024. Georgian tax law,
deadlines, penalties, and regulations may have changed since then, and you
do not know with certainty what changed unless the file search tool returns
a verified document confirming it.

STRICT BEHAVIOR for any question about a recent change, a new rule, a new
deadline, a new penalty amount, or "what changed in 2025/2026":
1. FIRST check file search results. If a verified document confirms the
   answer, use it and cite it naturally (e.g. "according to current
   guidance...").
2. If file search returns NOTHING relevant, or the question is about
   something specific and recent that you are not certain is verified —
   DO NOT invent an answer. DO NOT guess a plausible-sounding change.
3. Instead, respond honestly along these lines (adapt naturally, don't
   sound robotic or repeat one template every time):
   - State clearly that your base knowledge goes up to mid-2024 and recent
     legislative changes are outside what you can confirm with certainty.
   - Give what you DO know confidently (the stable, unchanged fundamentals)
     if relevant.
   - Recommend they verify the exact current figure/rule on rs.ge or with
     a licensed accountant before relying on it for a real decision.
4. NEVER state a specific new percentage, new deadline, or new penalty
   number for 2025-2026 unless it came from file search results. A wrong
   number stated confidently is far more dangerous than admitting
   uncertainty — being honestly unsure is a sign of a trustworthy
   assistant, not a weak one.
5. This caution applies especially to: penalty percentages, filing
   deadline changes, pension contribution mechanics, Form 200 scope,
   labor registration timing, threshold amounts (VAT/Small Business/Micro
   Business), and any newly named status or regime you're not fully sure
   still exists in its original form.

LANGUAGE RULE: Always respond in the SAME language the user writes in.
- If user writes in Georgian → respond in Georgian
- If user writes in English → respond in English
- If user writes in Russian → respond in Russian
- Never switch languages unless user explicitly asks

Be professional, warm, and concise. Never be vague — give specific, actionable answers.
If a concept is unclear to the user and they ask for clarification, explain it
in very simple terms with a concrete real-life example.

## MEMORY MANAGEMENT
When the user shares business facts (in ANY message), extract and save them:

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
SAVE: language preference — Georgian

Always extract facts even from casual conversation:
- "ჩვენ 5 თანამშრომელი გვყავს" → SAVE: employees — 5
- "I speak English" → SAVE: language preference — English
- "бизнес в Тбилиси" → SAVE: location — Tbilisi, Georgia

If the user CORRECTS a previously saved fact:
UPDATE: [old fact keyword] → [new fact]

## ⚠️ CRITICAL ONBOARDING RULES
RULE 1: Send EXACTLY ONE question per message. ONE. Never two. Never three.
RULE 2: Do NOT combine questions with "და", "ასევე", "and", "also", "также".
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
            CREATE TABLE IF NOT EXISTS monthly_tracker (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                year       INTEGER NOT NULL,
                month      INTEGER NOT NULL,
                income     REAL    DEFAULT 0,
                expenses   REAL    DEFAULT 0,
                tax        REAL    DEFAULT 0,
                notes      TEXT    DEFAULT '',
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, year, month)
            );
            CREATE INDEX IF NOT EXISTS idx_conv_user    ON conversations(user_id);
            CREATE INDEX IF NOT EXISTS idx_tracker_user ON monthly_tracker(user_id);
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


def get_tax_regime(user_id: int) -> str:
    with get_db() as db:
        row = db.execute(
            "SELECT fact FROM memories WHERE user_id=? AND mem_key LIKE '%tax%regime%'",
            (user_id,)
        ).fetchone()
    return row["fact"] if row else ""


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


# ─── Monthly tracker helpers ──────────────────────────────────────────────────

def get_tracker_entry(user_id: int, year: int, month: int) -> dict:
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM monthly_tracker WHERE user_id=? AND year=? AND month=?",
            (user_id, year, month)
        ).fetchone()
    if row:
        return dict(row)
    return {"income": 0, "expenses": 0, "tax": 0, "notes": ""}


def save_tracker_entry(user_id: int, year: int, month: int,
                       income: float, expenses: float, tax: float, notes: str = ""):
    with get_db() as db:
        db.execute(
            """INSERT INTO monthly_tracker(user_id, year, month, income, expenses, tax, notes, updated_at)
               VALUES(?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(user_id, year, month) DO UPDATE SET
                   income=excluded.income, expenses=excluded.expenses,
                   tax=excluded.tax, notes=excluded.notes,
                   updated_at=excluded.updated_at""",
            (user_id, year, month, income, expenses, tax, notes)
        )


def get_tracker_summary(user_id: int, months: int = 6) -> list[dict]:
    with get_db() as db:
        rows = db.execute(
            """SELECT year, month, income, expenses, tax, notes
               FROM monthly_tracker WHERE user_id=?
               ORDER BY year DESC, month DESC LIMIT ?""",
            (user_id, months)
        ).fetchall()
    return [dict(r) for r in rows]


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


# ─── OpenAI calls ─────────────────────────────────────────────────────────────

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
    return extract_and_clean(user_id, response.output_text.strip())


# ─── Export helper ────────────────────────────────────────────────────────────

# ─── Tracker keyboard ─────────────────────────────────────────────────────────

def tracker_keyboard(year: int, month: int):
    month_names = ["იანვ", "თებ", "მარტ", "აპრ", "მაი", "ივნ",
                   "ივლ", "აგვ", "სექ", "ოქტ", "ნოე", "დეკ"]
    prev_month = month - 1 if month > 1 else 12
    prev_year  = year if month > 1 else year - 1
    next_month = month + 1 if month < 12 else 1
    next_year  = year if month < 12 else year + 1

    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(f"◀ {month_names[prev_month-1]}", callback_data=f"tracker:{prev_year}:{prev_month}"),
            InlineKeyboardButton(f"{month_names[month-1]} {year}", callback_data="tracker_noop"),
            InlineKeyboardButton(f"{month_names[next_month-1]} ▶", callback_data=f"tracker:{next_year}:{next_month}"),
        ],
        [InlineKeyboardButton("✏️ შეყვანა", callback_data=f"tracker_edit:{year}:{month}")],
        [InlineKeyboardButton("« მენიუ",    callback_data="back_to_menu")],
    ])


# ─── Keyboards ────────────────────────────────────────────────────────────────

def main_menu_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📋 ჩემი ინფო",   callback_data="show_memories"),
            InlineKeyboardButton("✏️ რედაქტირება",  callback_data="edit_memories"),
        ],
        [
            InlineKeyboardButton("📊 ტრეკერი",      callback_data="show_tracker"),
            InlineKeyboardButton("📅 ვადები",        callback_data="show_deadlines"),
        ],
        [
            InlineKeyboardButton("🔄 კითხვარი",     callback_data="switch_menu"),
        ],
        [
            InlineKeyboardButton("🗑 ინფო წაშლა",   callback_data="forget"),
            InlineKeyboardButton("🔃 ისტ. წაშლა",   callback_data="reset"),
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


def format_tracker_text(user_id: int, year: int, month: int) -> str:
    month_names = ["იანვარი", "თებერვალი", "მარტი", "აპრილი", "მაისი", "ივნისი",
                   "ივლისი", "აგვისტო", "სექტემბერი", "ოქტომბერი", "ნოემბერი", "დეკემბერი"]
    entry  = get_tracker_entry(user_id, year, month)
    profit = entry["income"] - entry["expenses"] - entry["tax"]

    text = (
        f"📊 *{month_names[month-1]} {year}*\n\n"
        f"💚 შემოსავალი: *{entry['income']:,.2f} ₾*\n"
        f"🔴 ხარჯი: *{entry['expenses']:,.2f} ₾*\n"
        f"🟡 გადასახადი: *{entry['tax']:,.2f} ₾*\n"
        f"────────────────\n"
        f"💵 მოგება: *{profit:,.2f} ₾*\n"
    )
    if entry.get("notes"):
        text += f"\n📝 {entry['notes']}"
    return text


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
        "/tracker — ყოველთვიური ფინანსური ტრეკერი\n"
        "/switch — კითხვარის შეცვლა\n"
        "/deadlines — საგადასახადო ვადები\n"
        "/forget — ინფოს წაშლა\n"
        "/reset — საუბრის ისტორიის წაშლა\n\n"
        "გამომიგზავნე PDF, Word, ფოტო ან rs.ge სქრინშოტი (მაქს. 5MB)\n\n"
        "გამომიგზავნე ხმოვანი შეტყობინება — ვისმენ და ვპასუხობ\n\n"
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


async def cmd_tracker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    now     = datetime.now()
    text    = format_tracker_text(user_id, now.year, now.month)
    await update.message.reply_text(
        text,
        parse_mode="Markdown",
        reply_markup=tracker_keyboard(now.year, now.month),
    )


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
    user_id   = update.message.from_user.id
    regime    = get_tax_regime(user_id)
    deadlines = get_upcoming_deadlines(days_ahead=30, tax_regime=regime)
    if not deadlines:
        await update.message.reply_text("✅ მომავალ 30 დღეში საგადასახადო ვადები არ არის.")
    else:
        text = "📅 *მომავალი საგადასახადო ვადები:*\n\n" + "\n".join(deadlines)
        await update.message.reply_text(text, parse_mode="Markdown")


# ─── Document handler ─────────────────────────────────────────────────────────

# ─── Voice handler ────────────────────────────────────────────────────────────

# ─── Callback query handler ───────────────────────────────────────────────────

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query   = update.callback_query
    user_id = query.from_user.id
    data    = query.data
    await query.answer()

    # ── Reset ─────────────────────────────────────────────────────────────────
    if data == "confirm_reset":
        await query.edit_message_text("⏳ იშლება...")
        await do_full_reset(user_id, update)
        return

    if data == "cancel_reset":
        await query.edit_message_text("❌ გაუქმდა. ყველაფერი ისევ ადგილზეა ✅")
        return

    # ── Edit fact ─────────────────────────────────────────────────────────────
    if data.startswith("del_fact:"):
        mem_key  = data[9:]
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

    # ── Tracker ───────────────────────────────────────────────────────────────
    if data == "show_tracker":
        now  = datetime.now()
        text = format_tracker_text(user_id, now.year, now.month)
        await query.edit_message_text(
            text, parse_mode="Markdown",
            reply_markup=tracker_keyboard(now.year, now.month)
        )
        return

    if data.startswith("tracker:"):
        _, year, month = data.split(":")
        text = format_tracker_text(user_id, int(year), int(month))
        await query.edit_message_text(
            text, parse_mode="Markdown",
            reply_markup=tracker_keyboard(int(year), int(month))
        )
        return

    if data == "tracker_noop":
        return

    if data.startswith("tracker_edit:"):
        _, year, month = data.split(":")
        month_names = ["იანვარი", "თებერვალი", "მარტი", "აპრილი", "მაისი", "ივნისი",
                       "ივლისი", "აგვისტო", "სექტემბერი", "ოქტომბერი", "ნოემბერი", "დეკემბერი"]
        context.user_data["tracker_edit"] = {"year": int(year), "month": int(month), "step": "income"}
        await query.edit_message_text(
            f"✏️ *{month_names[int(month)-1]} {year} — შეყვანა*\n\n"
            f"შემოსავალი რამდენი იყო? (ლარში, მხოლოდ რიცხვი)\n"
            f"მაგ: `5000`",
            parse_mode="Markdown",
        )
        return

    # ── Onboarding ────────────────────────────────────────────────────────────
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

    # ── Switch ────────────────────────────────────────────────────────────────
    if data == "switch_full":
        set_onboarding_state(user_id, "full")
        clear_history(user_id)
        await query.edit_message_text("📋 *სრულ კითხვარზე გადავედი*\n\nშენი ინფო შენახულია ✅", parse_mode="Markdown")
        await run_ai_and_reply(update, context, user_id,
            "User switched to detailed onboarding. Previous facts are saved. Ask the next missing question only. ONE question.",
            mode="full", save_user_msg=False)
        return

    if data == "switch_quick":
        set_onboarding_state(user_id, "quick")
        clear_history(user_id)
        await query.edit_message_text("⚡ *სწრაფ კითხვარზე გადავედი*\n\nშენი ინფო შენახულია ✅", parse_mode="Markdown")
        await run_ai_and_reply(update, context, user_id,
            "User switched to quick onboarding. Previous facts are saved. Ask the next missing question only. ONE question.",
            mode="quick", save_user_msg=False)
        return

    # ── Menu ──────────────────────────────────────────────────────────────────
    if data == "show_memories":
        facts = load_memories(user_id)
        if not facts:
            await query.edit_message_text("ჯერ არაფერი შენახული მაქვს.")
        else:
            lines = "\n".join(f"• {f}" for f in facts)
            await query.edit_message_text(f"📋 *შენი ბიზნეს ინფო:*\n\n{lines}", parse_mode="Markdown", reply_markup=main_menu_keyboard())
        return

    if data == "edit_memories":
        memories = load_memories_with_keys(user_id)
        if not memories:
            await query.edit_message_text("ჯერ არაფერი შენახული მაქვს.")
        else:
            await query.edit_message_text("✏️ *რედაქტირება*\n\nდააჭირე ფაქტს წასაშლელად:", parse_mode="Markdown", reply_markup=edit_list_keyboard(memories))
        return

    if data == "show_deadlines":
        regime    = get_tax_regime(user_id)
        deadlines = get_upcoming_deadlines(days_ahead=30, tax_regime=regime)
        text      = ("📅 *მომავალი საგადასახადო ვადები:*\n\n" + "\n".join(deadlines)) if deadlines else "✅ მომავალ 30 დღეში ვადები არ არის."
        await query.edit_message_text(text, parse_mode="Markdown", reply_markup=main_menu_keyboard())
        return

    if data == "switch_menu":
        await query.edit_message_text("🔄 *კითხვარის შეცვლა*\n\nშენი შენახული ინფო *არ წაიშლება*.", parse_mode="Markdown", reply_markup=switch_keyboard())
        return

    if data == "forget":
        await query.edit_message_text("⚠️ *დარწმუნებული ხარ?*\n\nყველა შენახული ინფო წაიშლება.", parse_mode="Markdown", reply_markup=confirm_reset_keyboard())
        return

    if data == "reset":
        clear_history(user_id)
        await query.edit_message_text("✅ საუბრის ისტორია წაიშალა.", reply_markup=main_menu_keyboard())
        return

    if data == "back_to_menu":
        ob_state = get_onboarding_state(user_id)
        if ob_state == "done":
            await query.edit_message_text("რით შეგიძლია დაგეხმარო?", reply_markup=main_menu_keyboard())
        else:
            await send_onboarding_choice(update, edit=True)
        return


# ─── Message handler ──────────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id   = update.message.from_user.id
    user_text = update.message.text.strip()

    # ── Tracker data entry ────────────────────────────────────────────────────
    tracker_edit = context.user_data.get("tracker_edit")
    if tracker_edit:
        step = tracker_edit.get("step")
        year = tracker_edit["year"]
        month = tracker_edit["month"]

        try:
            value = float(user_text.replace(",", ".").replace(" ", ""))
        except ValueError:
            await update.message.reply_text("❌ მხოლოდ რიცხვი ჩაწერე, მაგ: `5000`", parse_mode="Markdown")
            return

        if step == "income":
            tracker_edit["income"] = value
            tracker_edit["step"]   = "expenses"
            context.user_data["tracker_edit"] = tracker_edit
            await update.message.reply_text("ხარჯი რამდენი იყო? (ლარში)\nმაგ: `2000`", parse_mode="Markdown")
        elif step == "expenses":
            tracker_edit["expenses"] = value
            tracker_edit["step"]     = "tax"
            context.user_data["tracker_edit"] = tracker_edit
            await update.message.reply_text("გადასახადი რამდენი გადაიხადე? (ლარში)\nმაგ: `50`", parse_mode="Markdown")
        elif step == "tax":
            save_tracker_entry(
                user_id, year, month,
                tracker_edit.get("income", 0),
                tracker_edit.get("expenses", 0),
                value,
            )
            context.user_data.pop("tracker_edit", None)
            text = format_tracker_text(user_id, year, month)
            await update.message.reply_text(
                f"✅ შენახულია!\n\n{text}",
                parse_mode="Markdown",
                reply_markup=tracker_keyboard(year, month),
            )
        return

    # ── Reset trigger ─────────────────────────────────────────────────────────
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
    app.add_handler(CommandHandler("tracker",   cmd_tracker))
    app.add_handler(CommandHandler("forget",    cmd_forget))
    app.add_handler(CommandHandler("reset",     cmd_reset))
    app.add_handler(CommandHandler("switch",    cmd_switch))
    app.add_handler(CommandHandler("deadlines", cmd_deadlines))
    app.add_handler(CallbackQueryHandler(handle_callback))
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
