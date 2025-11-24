from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import os

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

# Load  DB 
DB_PATH = "db.json"
DB = {}
if os.path.exists(DB_PATH):
    with open(DB_PATH, "r", encoding="utf-8") as f:
        DB = json.load(f)

# --------- language + greeting texts ----------

SUPPORTED_LANGS = {"en", "hi"}   # add more later if needed

GREET_KEYWORDS = {
    "en": ["hi", "hello", "hey", "hai"],
    "hi": ["namaste", "namaskar", "namasthe", "नमस्ते", "नमस्कार"],
}

GREET_MESSAGE = {
    "en": (
        "👋 Hello! I'm *Ziva*, your whatsapp health assistant.\n\n"
    ),
    "hi": (
        "👋 नमस्ते! मैं *Ziva* हूँ, आपका ऑफ़लाइन स्वास्थ्य सहायक।\n\n"
        "आप ऐसे सवाल पूछ सकते हैं:\n"
        "• डेंगू के लक्षण\n"
        "• मलेरिया से बचाव कैसे करें\n"
        "• शिशु / बच्चा / वयस्क / गर्भवती के टीकाकरण की सूची\n\n"
        "किसी भाषा को चुनने के लिए आप ऐसे लिख सकते हैं:\n"
        "`lang:hi डेंगू के लक्षण`\n\n"
        "आपात स्थिति में तुरंत डॉक्टर या नज़दीकी अस्पताल से संपर्क करें।"
    ),
}

FALLBACK_MESSAGE = {
    "en": (
        "I couldn't find an  answer for that.\n\n"
        "Searching the web.....\n"
    ),
    "hi": (
        "मुझे इस प्रश्न का ऑफ़लाइन उत्तर नहीं मिला।\n\n"
        "आप इन विषयों पर पूछ सकते हैं:\n"
        "• डेंगू / मलेरिया / टीबी के लक्षण\n"
        "• मलेरिया / टाइफ़ॉइड से बचाव कैसे करें\n"
        "• शिशु / बच्चा / वयस्क / गर्भवती के टीकाकरण की सूची\n\n"
        "छाती में दर्द, तेज़ रक्तस्राव या सांस लेने में दिक्कत जैसी आपात स्थिति में "
        "कृपया तुरंत डॉक्टर या अस्पताल से संपर्क करें।"
    ),
}


class ChatMessage(BaseModel):
    message: str
    lang: str = "en"


# ------------------ core logic ------------------

def search_db(text: str, lang: str = "en") -> str | None:
    """
    Search the disease JSON in db.json.

    Expected format of db.json:

    {
      "dengue": {
        "en": {
          "what": "....",
          "symptoms": [...],
          "prevention": [...],
          "remedies": [...]
        },
        "hi": { ... }
      },
      "malaria": { ... }
    }

    Steps:
    1) Detect disease name mentioned in the user text.
    2) Detect what user is asking: what / symptoms / prevention / remedies.
    3) Return a nicely formatted string for WhatsApp.
    """
    if not DB or not text:
        return None

    q = text.lower().strip()
    lang = lang if lang in SUPPORTED_LANGS else "en"

    # ---------- 1) detect disease ----------
    disease_key = None
    for d in DB.keys():
        d_low = d.lower()
        if d_low in q:
            disease_key = d
            break

    if not disease_key:
        # no disease name matched
        return None

    disease_data = DB.get(disease_key, {})
    lang_block = disease_data.get(lang) or disease_data.get("en")
    if not lang_block:
        return None

    # ---------- 2) detect what user is asking (category) ----------
    category = "what"   # default if nothing specific found

    # English keywords
    if any(w in q for w in ["symptom", "symptoms", "signs"]):
        category = "symptoms"
    elif any(w in q for w in ["prevent", "prevention", "avoid", "protection"]):
        category = "prevention"
    elif any(w in q for w in ["remedy", "remedies", "treat", "treatment", "cure"]):
        category = "remedies"

    # Hindi keywords
    if any(w in q for w in ["लक्षण"]):
        category = "symptoms"
    elif any(w in q for w in ["बचाव", "रोकथाम"]):
        category = "prevention"
    elif any(w in q for w in ["उपचार", "इलाज"]):
        category = "remedies"

    data = lang_block.get(category)
    if data is None:
        # fallback to "what" if present
        data = lang_block.get("what")
        if data is None:
            return None
        category = "what"

    # ---------- 3) format answer for WhatsApp ----------
    title_map = {
        "what": "About",
        "symptoms": "Symptoms",
        "prevention": "Prevention",
        "remedies": "Remedies"
    }

    if lang == "hi":
        title_map = {
            "what": "क्या है",
            "symptoms": "लक्षण",
            "prevention": "बچाव",
            "remedies": "उपचार / घरेलू उपाय"
        }

    heading = title_map.get(category, category.capitalize())
    disease_title = disease_key.title()

    if isinstance(data, list):
        bullet_lines = "\n".join(f"• {item}" for item in data)
        return f"*{disease_title} – {heading}*\n{bullet_lines}"
    else:
        return f"*{disease_title} – {heading}*\n{data}"


def process_message(text: str, lang: str = "en") -> dict:
    """
    Single place where we decide the answer.
    Used by both /chat (web) and /whatsapp (Twilio).
    Returns: {"type": ..., "answer": "...", "extra": optional}
    """
    text = (text or "").strip()
    lang = (lang or "en").lower()
    if lang not in SUPPORTED_LANGS:
        lang = "en"

    lower = text.lower()

    if not text:
        return {
            "type": "fallback",
            "answer": GREET_MESSAGE.get(lang, GREET_MESSAGE["en"])
        }

    # 0) Greetings (hi, hello, namaste, etc.)
    for g in GREET_KEYWORDS.get(lang, []):
        if lower == g or lower.startswith(g + " "):
            return {
                "type": "greeting",
                "answer": GREET_MESSAGE.get(lang, GREET_MESSAGE["en"])
            }

    # 0.5) Thank you messages
    THANK_KEYWORDS = {
        "en": ["thank", "thanks", "thank you", "thx", "ty"],
        "hi": ["dhanyavad", "धन्यवाद", "shukriya", "शुक्रिया"]
    }

    for t in THANK_KEYWORDS.get(lang, []):
        if t in lower:
            thank_reply = {
                "en": "😊 You're welcome! Let me know if you need more health information.",
                "hi": "😊 आपका स्वागत है! यदि आपको और स्वास्थ्य जानकारी चाहिए, तो बताएं।"
            }
            return {
                "type": "thanks",
                "answer": thank_reply.get(lang, thank_reply["en"])
            }

    # 1) FAQ (40-disease database using new JSON format)
    db_answer = search_db(text, lang)
    if db_answer:
        return {"type": "faq", "answer": db_answer}

    # 2) Vaccination schedule
    if any(w in lower for w in [
        "vaccine", "vaccination", "vaccine schedule", "immunization",
        "टीका", "टीकाकरण"
    ]):
        return {
            "type": "vaccination",
            "answer": "Here is the offline vaccination schedule (infant, child, adult, pregnant).",
            "extra": vaccinations.VACCINATION_SCHEDULES
        }

    # 3) Preventive health modules
    for key in preventive_health.MODULES.keys():
        if key in lower:
            return {
                "type": "preventive",
                "answer": preventive_health.MODULES[key]
            }

    # 4) Simple disease info module (if you still keep this separate)
    disease_info = diseases_multilang.find_disease(text, lang)
    if disease_info:
        return {"type": "disease", "answer": disease_info}

    # 5) Fallback
    fallback_text = FALLBACK_MESSAGE.get(lang, FALLBACK_MESSAGE["en"])
    return {"type": "fallback", "answer": fallback_text}


# ------------------ web UI route ------------------

@app.get("/")
def index():
    path = os.path.join("public", "index.html")
    if os.path.exists(path):
        return HTMLResponse(open(path, "r", encoding="utf-8").read())
    return HTMLResponse("<h1>MedXpert</h1><p>UI not found.</p>")


# ------------------ /chat (web) ------------------

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


# ------------------ /whatsapp (Twilio webhook) ------------------

@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    """
    Twilio will call this URL when someone sends a WhatsApp message.
    We read 'Body', parse optional lang prefix, call process_message,
    and reply with TwiML XML.
    """
    form = await request.form()
    raw_body = (form.get("Body") or "").strip()

    lang = "en"
    text = raw_body

    # Parse language prefix, e.g. "lang:hi डेंगू के लक्षण"
    lower = raw_body.lower()
    if lower.startswith("lang:"):
        rest = raw_body[5:].lstrip()  # after "lang:"
        if rest:
            first = rest.split()[0]           # e.g. "hi" or "hi;..."
            lang_code = first.split(";")[0].lower().strip()
            if lang_code in SUPPORTED_LANGS:
                lang = lang_code
                text = rest[len(first):].lstrip(" ;,")
            else:
                lang = "en"
                text = rest

    result = process_message(text, lang)
    answer = result.get("answer") or "Sorry, something went wrong."

    resp = MessagingResponse()
    resp.message(answer)

    return PlainTextResponse(content=str(resp), media_type="application/xml")
