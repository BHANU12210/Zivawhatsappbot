# diseases_multilang.py
# Simple offline multilingual disease lookup (basic sample)

DISEASES = {
    "covid": {
        "en": "COVID-19 is a viral illness affecting the respiratory system.",
        "hi": "COVID-19 एक श्वसन तंत्र को प्रभावित करने वाली बीमारी है।",
        "ta": "COVID-19 என்பது மூச்சுத்திணறல் அமைப்பை பாதிக்கும் நோய்.",
        "te": "COVID-19 శ్వాసకోశాన్ని ప్రభావితం చేసే వైరల్ వ్యాధి.",
        "kn": "COVID-19 ಉಸಿರಾಟವನ್ನು ಪ್ರಭಾವಿಸುವ ರೋಗ.",
        "ml": "COVID-19 ശ്വാസകോശത്തെ ബാധിക്കുന്ന വൈറസ് രോഗമാണ്."
    }
}

def find_disease(text, lang="en"):
    text = text.lower()
    for key in DISEASES:
        if key in text:
            return DISEASES[key].get(lang) or DISEASES[key].get("en")
    return None
