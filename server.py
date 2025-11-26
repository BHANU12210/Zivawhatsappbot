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

# ------------------ KOCHI HOSPITALS (NEW) ------------------
# Sources: Aster Medcity, Amrita AIMS, VPS Lakeshore, Rajagiri, Apollo Adlux, Lisie, etc.
# (These entries contain name, address, phone, url and tags — tags help pick hospitals for specific disease categories.)
HOSPITALS_KOCHI = [
    {
        "name": "Aster Medcity",
        "address": "Kuttisahib Road, Cheranelloor, Kochi, Kerala 682027",
        "phone": "+91-484-6699999",
        "url": "https://www.asterhospitals.in/hospitals/aster-medcity-kochi",
        "tags": ["multispecialty", "tertiary", "critical-care", "surgery"]
    },
    {
        "name": "Amrita Institute of Medical Sciences (AIMS) - Kochi",
        "address": "Amrita Lane, Elamakkara P.O., Kochi 682026",
        "phone": "+91-484-2802020",
        "url": "https://www.amritahospitals.org/kochi",
        "tags": ["multispecialty", "teaching-hospital", "critical-care", "organ-transplant"]
    },
    {
        "name": "VPS Lakeshore Hospital",
        "address": "Nettoor/Maradu, Ernakulam, Kochi 682040",
        "phone": "+91-484-2701032",
        "url": "https://www.vpslakeshorehospital.com",
        "tags": ["multispecialty", "cardiac", "critical-care", "emergency"]
    },
    {
        "name": "Rajagiri Hospital (Aluva)",
        "address": "Chunangamvely Road, GTN Junction, Aluva, Kochi",
        "phone": "+91-484-2905100",
        "url": "https://www.rajagirihospital.com",
        "tags": ["multispecialty", "nephrology", "transplant", "critical-care"]
    },
    {
        "name": "Apollo Adlux Hospital",
        "address": "NH 66, Kochi",
        "phone": "+91-484-2399000",
        "url": "https://www.apollohospitals.com",
        "tags": ["multispecialty", "critical-care", "emergency"]
    },
    {
        "name": "Lisie Hospital",
        "address": "Nadama, Kochi",
        "phone": "+91-484-2604626",
        "url": "https://www.lisiehospital.org",
        "tags": ["multispecialty", "cardiology", "surgery"]
    }
]

# Helper to pick hospitals relevant to a disease key
def get_hospitals_for_disease(disease_key: str, limit: int = 3) -> list[dict]:
    """
    Very simple relevance: match disease-key derived tags to hospital tags.
    If no tag match, return top multispecialty hospitals.
    """
    if not disease_key:
        return []

    # Build simple keyword -> tag map (extend as needed)
    keyword_tag_map = {
        "fever": ["emergency", "critical-care", "multispecialty"],
        "dengue": ["emergency", "multispecialty"],
        "malaria": ["emergency", "multispecialty"],
        "covid": ["emergency", "critical-care"],
        "cardiac": ["cardiac", "multispecialty"],
        "heart": ["cardiac"],
        "kidney": ["nephrology", "multispecialty", "transplant"],
        "cancer": ["oncology", "multispecialty"],
        "pregnancy": ["maternity", "multispecialty"],
        # fallback will match multispecialty
    }

    low = disease_key.lower()
    matched_tags = set()
    for kw, tags in keyword_tag_map.items():
        if kw in low:
            matched_tags.update(tags)

    # Score hospitals by tag overlap
    scored = []
    for h in HOSPITALS_KOCHI:
        score = len(matched_tags & set(h.get("tags", [])))
        # prefer hospitals marked multispecialty if no matches
        if not matched_tags and "multispecialty" in h.get("tags", []):
            score += 1
        scored.append((score, h))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [h for s, h in scored if s >= 0][:limit]
    return results

# ------------------ language + greeting texts ------------------

SUPPORTED_LANGS = {"en"}

GREET_KEYWORDS = {
    "en": ["hi", "hello", "hey", "hai"],
}

# HUMANIZED GREETING (NEW: emoji + friendlier tone)
GREET_MESSAGE = {
    "en": (
        "👋 Hey — I'm *WellnessHelp*, your friendly WhatsApp health assistant!\n\n"
        "Tell me your symptoms (e.g., \"fever and body ache\") or ask about a disease like \"dengue\".\n"
        "I can give: symptoms, prevention, remedies, vaccination info, and local hospital suggestions in Kochi."
    ),
}

FALLBACK_MESSAGE = {
    "en": (
        "😕 I couldn't find an exact answer for that.\n\n"
        "Try asking about:\n"
        "• Dengue symptoms\n"
        "• Malaria prevention\n"
        "• Vaccination schedule\n\n"
        "If this is an emergency, please contact local emergency services or visit the nearest hospital right away."
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

    # If matched by NLP → short diagnosis-like message (HUMANISED)
    if not matched_by_key:
        return (
            f"🤖 It looks like your symptoms match *{disease_title}*.\n"
            f"⚠️ This is only an initial guess — please consult a doctor if you feel unwell.\n"
            f"I can also suggest nearby hospitals in Kochi if you'd like."
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
        return f"💡 *{disease_title} – {heading}*\n{bullet_lines}"
    else:
        return f"💡 *{disease_title} – {heading}*\n{data}"

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
            "answer": "😊 You're welcome! Glad I could help. Anything else I can do?"
        }

    # FAQ
    db_answer = search_db(text, lang)
    if db_answer:
        # Determine disease_key again so we can attach hospitals if relevant
        disease_key = None
        for d in DB.keys():
            variants = {d.lower(), d.lower().replace("_", " "), d.lower().replace("-", " ")}
            if any(v in lower for v in variants):
                disease_key = d
                break
        if not disease_key:
            disease_key = nlp_guess_disease(text)

        payload = {"answer": db_answer}
        if disease_key:
            # Attach hospital suggestions (humanized)
            hospitals = get_hospitals_for_disease(disease_key, limit=3)
            hosp_text_lines = []
            for h in hospitals:
                hosp_text_lines.append(f"• {h['name']} — {h['address']} (☎ {h['phone']})")
            hosp_block = "\n".join(hosp_text_lines)
            payload["hospitals"] = hospitals
            payload["answer"] += "\n\n🏥 *Nearby hospitals in Kochi you can consider:*\n" + hosp_block
        return {"type": "db", "answer": payload}

    # Vaccination
    if any(w in lower for w in ["vaccine", "vaccination", "immunization"]):
        return {
            "type": "vaccination",
            "answer": "💉 Here is the vaccination schedule (infant, child, adult).",
            "extra": vaccinations.VACCINATION_SCHEDULES
        }

    # Preventive health modules
    for key in preventive_health.MODULES.keys():
        if key in lower:
            return {"type": "preventive", "answer": preventive_health.MODULES[key]}

    # Older simple-disease module
    disease_info = diseases_multilang.find_disease(text, "en")
    if disease_info:
        # If this module returns data, also suggest hospitals
        guessed = nlp_guess_disease(text)  # try to map to DB key if possible
        hospitals = []
        if guessed:
            hospitals = get_hospitals_for_disease(guessed, limit=3)
        answer_text = "🤝 " + disease_info
        payload = {"answer": answer_text}
        if hospitals:
            hosp_text_lines = [f"• {h['name']} — {h['address']} (☎ {h['phone']})" for h in hospitals]
            payload["hospitals"] = hospitals
            payload["answer"] += "\n\n🏥 *Nearby hospitals in Kochi you can consider:*\n" + "\n".join(hosp_text_lines)
        return {"type": "disease", "answer": payload}

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
    # result["answer"] may be a dict payload now (for db/disease types)
    if isinstance(result["answer"], dict):
        payload = result["answer"]
        return JSONResponse({
            "type": result["type"],
            "payload": {
                "answer": payload.get("answer"),
                "extra": payload.get("extra"),
                "hospitals": payload.get("hospitals")
            }
        })
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
    # handle the new payload structure for WhatsApp: send only the friendly text reply
    answer = ""
    if isinstance(result.get("answer"), dict):
        answer = result["answer"].get("answer") or "Sorry, something went wrong."
    else:
        answer = result.get("answer") or "Sorry, something went wrong."

    resp = MessagingResponse()
    resp.message(answer)

    return PlainTextResponse(content=str(resp), media_type="application/xml")
