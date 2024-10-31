import pandas as pd
import pandas as pd  # For data loading and manipulation
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and validation sets
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments  # Hugging Face Transformers for BERT model, tokenizer, trainer, and training arguments

# Load the dataset
df = pd.read_csv('cardio_train.csv', delimiter=';')

# Prepare the text and labels
df['text'] = "Age: " + df['age'].astype(str) + \
             ", Gender: " + df['gender'].astype(str) + \
             ", Height: " + df['height'].astype(str) + " cm" + \
             ", Weight: " + df['weight'].astype(str) + " kg" + \
             ", Systolic BP: " + df['ap_hi'].astype(str) + \
             ", Diastolic BP: " + df['ap_lo'].astype(str) + \
             ", Cholesterol: " + df['cholesterol'].astype(str) + \
             ", Glucose: " + df['gluc'].astype(str) + \
             ", Smoker: " + df['smoke'].astype(str) + \
             ", Alcohol: " + df['alco'].astype(str) + \
             ", Active: " + df['active'].astype(str)

# Labels for the model
df['label'] = df['cardio']
from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)
import torch

class CardioDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
tokenizer = BertTokenizer.from_pretrained("medicalai/ClinicalBERT")
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

train_dataset = CardioDataset(train_encodings, train_labels)
val_dataset = CardioDataset(val_encodings, val_labels)
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

model = BertForSequenceClassification.from_pretrained("medicalai/ClinicalBERT", num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy='epoch'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
# Save model state_dict
torch.save(model.state_dict(), 'fine_tuned_clinicalbert.pth')
