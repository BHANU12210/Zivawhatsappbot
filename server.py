from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import os
import re
from collections import Counter

from dotenv import load_dotenv
load_dotenv()

# Twilio for WhatsApp replies
from twilio.twiml.messaging_response import MessagingResponse

# Local modules
import vaccinations
import preventive_health
import diseases_multilang
import languages

# ------------------ basic setup ------------------

app = FastAPI()
if os.path.isdir("public"):
    app.mount("/public", StaticFiles(directory="public"), name="public")

# Load FAQ DB 
DB_PATH = "db.json"
DB = {}
if os.path.exists(DB_PATH):
    with open(DB_PATH, "r", encoding="utf-8") as f:
        DB = json.load(f)

print("DB loaded keys:", list(DB.keys()))

# ------------------ simple NLP index over FAQ ------------------

STOP_WORDS = {
    "a", "an", "the", "is", "are", "am", "i", "you", "he", "she", "it", "we", "they",
    "of", "for", "to", "in", "on", "and", "or", "but", "with", "at", "from",
    "this", "that", "these", "those", "about", "what", "how", "when", "why",
    "do", "does", "did", "my", "your", "his", "her", "their", "our", "have",
    "has", "had", "me", "be", "been", "was", "were"
}

DISEASE_DOCS: dict[str, set[str]] = {}

def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in STOP_WORDS]

def build_disease_index() -> None:
    global DISEASE_DOCS
    DISEASE_DOCS = {}

    if not DB:
        return

    for disease_key, disease_data in DB.items():
        lang_block = disease_data.get("en")
        if not lang_block:
            continue

        parts = []
        for field in ("what", "symptoms", "prevention", "remedies"):
            val = lang_block.get(field)
            if isinstance(val, list):
                parts.extend(val)
            elif isinstance(val, str):
                parts.append(val)

        if not parts:
            continue

        doc = " ".join(parts)
        tokens = _tokenize(doc)
        if tokens:
            DISEASE_DOCS[disease_key] = set(tokens)

    print("DISEASE_DOCS built for:", list(DISEASE_DOCS.keys()))

def nlp_guess_disease(text: str) -> str | None:
    if not DISEASE_DOCS:
        return None

    tokens = set(_tokenize(text))
    if not tokens:
        return None

    best_key = None
    best_overlap = 0

    for disease_key, doc_tokens in DISEASE_DOCS.items():
        overlap = len(tokens & doc_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_key = disease_key

    if best_overlap < 3:
        return None

    print("[NLP] Guessed disease", best_key, "overlap:", best_overlap)
    return best_key

build_disease_index()

# ------------------ language + greeting texts ------------------

SUPPORTED_LANGS = {"en"}

GREET_KEYWORDS = {
    "en": ["hi", "hello", "hey", "hai"],
}

GREET_MESSAGE = {
    "en": (
        "👋 Hello! I'm *Ziva*, your WhatsApp health assistant.\n\n"
        "You can ask about symptoms, prevention, remedies, or vaccination schedules."
    ),
}

FALLBACK_MESSAGE = {
    "en": (
        "I couldn't find an answer for that.\n\n"
        "Try asking about:\n"
        "• Dengue symptoms\n"
        "• Malaria prevention\n"
        "• Vaccination schedule\n\n"
        "If this is an emergency, contact a doctor immediately."
    )
}

class ChatMessage(BaseModel):
    message: str
    lang: str = "en"

# ------------------ UPDATED search_faq FUNCTION ------------------

def search_db(text: str, lang: str = "en") -> str | None:
    """
    English-only.
    If disease matched by NLP → short caution reply.
    If disease explicitly typed → full details.
    """
    if not DB or not text:
        return None

    q_original = text
    q = text.lower().strip()

    disease_key = None
    matched_by_key = False
    # Direct disease name match
    for d in DB.keys():
        d_low = d.lower()
        variants = {
            d_low,
            d_low.replace("_", " "),
            d_low.replace("-", " "),
        }
        if any(v in q for v in variants):
            disease_key = d
            matched_by_key = True
            break

    # NLP guess if name not explicitly mentioned
    if not disease_key:
        disease_key = nlp_guess_disease(q_original)

    if not disease_key:
        return None

    lang_block = DB[disease_key].get("en")
    if not lang_block:
        return None

    disease_title = disease_key.replace("_", " ").replace("-", " ").title()

    # If matched by NLP → short diagnosis-like message
    if not matched_by_key:
        return (
            f"You are showing symptoms that match *{disease_title}*.\n"
            f"It is more likely to be *{disease_title}*.\n"
            f"Consult a doctor if things get worse."
        )

    # User typed the disease → full details
    category = "what"

    if any(w in q for w in ["symptom", "symptoms", "signs"]):
        category = "symptoms"
    elif any(w in q for w in ["prevent", "prevention", "avoid"]):
        category = "prevention"
    elif any(w in q for w in ["treat", "treatment", "remedy", "remedies", "cure"]):
        category = "remedies"

    data = lang_block.get(category)
    if data is None:
        data = lang_block.get("what")
        category = "what"

    title_map = {
        "what": "About",
        "symptoms": "Symptoms",
        "prevention": "Prevention",
        "remedies": "Remedies",
    }

    heading = title_map.get(category, category.capitalize())

    if isinstance(data, list):
        bullet_lines = "\n".join("• " + item for item in data)
        return f"*{disease_title} – {heading}*\n{bullet_lines}"
    else:
        return f"*{disease_title} – {heading}*\n{data}"

# ------------------ main logic ------------------

def process_message(text: str, lang: str = "en") -> dict:
    text = (text or "").strip()
    lang = "en"
    lower = text.lower()

    if not text:
        return {"type": "fallback", "answer": GREET_MESSAGE["en"]}

    # Greetings
    for g in GREET_KEYWORDS["en"]:
        if lower == g or lower.startswith(g + " "):
            return {"type": "greeting", "answer": GREET_MESSAGE["en"]}

    # Thank-you replies
    THANKS = ["thank", "thanks", "thank you", "thx", "ty"]
    if any(t in lower for t in THANKS):
        return {
            "type": "thanks",
            "answer": "😊 You're welcome! Let me know if you need more health information."
        }

    # FAQ
    db_answer = search_db(text, lang)
    if db_answer:
        return {"type": "db", "answer": db_answer}

    # Vaccination
    if any(w in lower for w in ["vaccine", "vaccination", "immunization"]):
        return {
            "type": "vaccination",
            "answer": "Here is the vaccination schedule (infant, child, adult).",
            "extra": vaccinations.VACCINATION_SCHEDULES
        }

    # Preventive health modules
    for key in preventive_health.MODULES.keys():
        if key in lower:
            return {"type": "preventive", "answer": preventive_health.MODULES[key]}

    # Older simple-disease module
    disease_info = diseases_multilang.find_disease(text, "en")
    if disease_info:
        return {"type": "disease", "answer": disease_info}

    return {"type": "fallback", "answer": FALLBACK_MESSAGE["en"]}

# ------------------ web UI ------------------

@app.get("/")
def index():
    path = os.path.join("public", "index.html")
    if os.path.exists(path):
        return HTMLResponse(open(path, "r", encoding="utf-8").read())
    return HTMLResponse("<h1>Ziva</h1><p>UI not found.</p>")

# ------------------ /chat ------------------

@app.post("/chat")
def chat(msg: ChatMessage):
    result = process_message(msg.message, msg.lang)
    return JSONResponse({
        "type": result["type"],
        "payload": {
            "answer": result["answer"],
            "extra": result.get("extra")
        }
    })
# ------------------ WhatsApp webhook ------------------

@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    form = await request.form()
    raw_body = (form.get("Body") or "").strip()

    lang = "en"
    text = raw_body

    # Parse lang prefix if any (ignored now since only English)
    lower = raw_body.lower()
    if lower.startswith("lang:"):
        text = raw_body.split(" ", 1)[-1]

    result = process_message(text, lang)
    answer = result.get("answer") or "Sorry, something went wrong."

    resp = MessagingResponse()
    resp.message(answer)

    return PlainTextResponse(content=str(resp), media_type="application/xml")
