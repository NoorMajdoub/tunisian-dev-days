import spacy
import re

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

def extract_health_info(input_string):
    # Define expected keys
    res=""
    keys = {
        "Age": None,
        "Anaemia": None,
        "Creatinine Phosphokinase": None,
        "Diabetes": None,
        "Ejection Fraction": None,
        "High Blood Pressure": None,
        "Platelets": None,
        "Serum Creatinine": None,
        "Serum Sodium": None,
        "Sex": None,
        "Smoking": None,
        "Time": None
    }

    # Process the input string with spaCy
    doc = nlp(input_string)

    # Use regex patterns to find relevant information
    patterns = {
        "Age": r"(\d+)\s*years?\s*old",
        "Anaemia": r"anaemia.*?(\d+)",
        "Creatinine Phosphokinase": r"creatinine phosphokinase.*?(\d+)",
        "Diabetes": r"(does have diabetes|has diabetes|diabetes).*?(\d*)",
        "Ejection Fraction": r"ejection fraction.*?(\d+)",
        "High Blood Pressure": r"high blood pressure.*?(\d+)",
        "Platelets": r"platelets.*?(\d+)",
        "Serum Creatinine": r"serum creatinine.*?(\d+\.?\d*)",
        "Serum Sodium": r"serum sodium.*?(\d+)",
        "Sex": r"\b(female|girl|woman)\b|\b(male|boy|man)\b",
        "Smoking": r"\b(smokes|does smoke)\b"
    }

    # Search for each key in the input string
    for key, pattern in patterns.items():
        match = re.search(pattern, input_string, re.IGNORECASE)
        if match:
            if key == "Sex":
                # Determine sex based on keywords
                if match.group(0).lower() in ["female", "girl", "woman"]:
                    keys[key] = 0
                else:
                    keys[key] = 1
            elif key == "Smoking":
                # Determine smoking status
                keys[key] = 1 if match.group(0).lower() in ["smokes", "does smoke"] else 0
            elif key == "Diabetes":
                # Determine diabetes status
                keys[key] = 1 if match.group(0) else 0
            else:
                keys[key] = match.group(1)

    for key, value in keys.items():

         res+=key+":"+str(value)+","

    return res.rstrip(',')


# Example usage
input_string = "The patient is 12 years old, does have diabetes, smokes, and has anaemia 1. and she is a girl"
processed_output = extract_health_info(input_string)
print(processed_output)
