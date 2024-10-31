import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Step 1: Prepare the Input Text
#new_sample = "Age: 60, Anaemia: none, Creatinine Phosphokinase: 200, Diabetes: 1, Ejection Fraction: 40, High Blood Pressure: 1, Platelets: 250000, Serum Creatinine: 1.5, Serum Sodium: 140, Sex: 1, Smoking: 1, Time: 10"
new_sample ="Age:12,Anaemia:1,Creatinine Phosphokinase:200,Diabetes:1,Ejection Fraction:25,High Blood Pressure:89,Platelets:8,Serum Creatinine:90,Serum Sodium:99,Sex:0,Smoking:1,Time:None"
# Step 2: Tokenize the Input Text
tokenizer = BertTokenizer.from_pretrained("medicalai/ClinicalBERT")
inputs = tokenizer(new_sample, truncation=True, padding=True, max_length=512, return_tensors='pt')

# Step 3: Load the Fine-tuned Model
model = BertForSequenceClassification.from_pretrained("medicalai/ClinicalBERT", num_labels=2)
model.load_state_dict(torch.load('clinicalbertmodel/fine_tuned_clinicalbert.pth'))
model.eval()  # Set to evaluation mode

# Step 4: Make Predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Step 5: Interpret the Output
probabilities = torch.softmax(logits, dim=-1)
predicted_class = torch.argmax(probabilities, dim=-1).item()

print(f"Predicted class: {predicted_class}")  # Output the predicted class
print(f"Probabilities: {probabilities}")  # Output the probabilities for each class
