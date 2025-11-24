DISEASES = {
    "covid": {
        "name": "COVID-19",
        "symptoms": [
            "fever",
            "dry cough",
            "tiredness",
            "loss of taste or smell"
        ],
        "prevention": [
            "wear a mask",
            "wash hands regularly",
            "maintain social distance",
            "vaccination"
        ],
        "treatment": "Supportive care. Seek medical attention if experiencing breathing difficulties."
    },

    "diabetes": {
        "name": "Diabetes",
        "symptoms": [
            "increased thirst",
            "frequent urination",
            "fatigue",
            "blurred vision"
        ],
        "prevention": [
            "healthy diet",
            "regular exercise",
            "maintain healthy weight"
        ],
        "treatment": "Insulin or oral medications as prescribed by doctors."
    },

    "malaria": {
        "name": "Malaria",
        "symptoms": [
            "fever",
            "chills",
            "headache",
            "sweating"
        ],
        "prevention": [
            "use mosquito nets",
            "remove stagnant water",
            "mosquito repellents"
        ],
        "treatment": "Antimalarial drugs prescribed by a healthcare provider."
    },

    "tuberculosis": {
        "name": "Tuberculosis (TB)",
        "symptoms": [
            "persistent cough",
            "weight loss",
            "night sweats",
            "fever"
        ],
        "prevention": [
            "BCG vaccination",
            "good ventilation",
            "avoid close contact with TB patients"
        ],
        "treatment": "Long-term antibiotic treatment as part of a DOTS program."
    },

    "dengue": {
        "name": "Dengue Fever",
        "symptoms": [
            "high fever",
            "joint pain",
            "rash",
            "severe headache"
        ],
        "prevention": [
            "avoid mosquito bites",
            "remove standing water",
            "use repellents"
        ],
        "treatment": "Rest, hydration. Seek medical care if symptoms worsen."
    }
}

def find_disease(keyword: str):
    """Simple keyword-based search for disease."""
    keyword = keyword.lower()
    for key, info in DISEASES.items():
        if key in keyword or info["name"].lower() in keyword:
            return info
    return None
