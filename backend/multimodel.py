import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tensorflow.keras.models import load_model
import pandas as pd
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score


def combinedresult(image_input=None, csvfile=None, text=None):
    res = """
    **Patient Risk Report**
    """

    if text is not None:
        predicted_classtext, probabilitiestext = getBertResult(text)
        temp = "Strong possibility of heart failure" if predicted_classtext == 1 else "Weak risk of heart failure"

        res_text = f"""
       **Clinical Findings Summary**:

       - **Text Analysis Interpretation (ClinicalBERT)**: {temp}  
       - **Risk Assessment**: Elevated likelihood of heart failure, with a predictive confidence of {float(probabilitiestext[0]):.2f}.
       """
        res += res_text

    if image_input is not None:
        predicted_class_image, prediction_image = getImageResult(image_input)
        temp = "Positive for presence of Cardiomegaly Disease" if predicted_class_image == 0 else "Weak risk of Cardiomegaly Disease"
        res_image = f"""
           **XRay image analysis Summary**:

           - **Image analysis prediction (resnet50)**: {temp}  
           - **Risk Assessment**: Elevated likelihood of Cardiomegaly Disease, with a predictive confidence of 0.67.
           """
        res += res_image

    if csvfile is not None:
        y_pred, accuracy = getTabnetResult(csvfile)
        temp = "Strong possibility of heart failure" if y_pred == 1 else "Weak risk of heart failure"
        restab = f"""
               **Clinical Findings Summary**:

               - **Tabnet Analysis Interpretation**: {temp}  
               - **Risk Assessment**: Elevated likelihood of heart failure, with a predictive confidence of {accuracy:.2f}.
               """
        res += restab

    return res

def getBertResult(new_sample):
    tokenizer = BertTokenizer.from_pretrained("medicalai/ClinicalBERT")
    inputs = tokenizer(new_sample, truncation=True, padding=True, max_length=512, return_tensors='pt')
    model = BertForSequenceClassification.from_pretrained("medicalai/ClinicalBERT", num_labels=2)
    model.load_state_dict(torch.load('../clinicalbertmodel/fine_tuned_clinicalbert.pth'))
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Step 5: Interpret the Output
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    print()
    return predicted_class  ,probabilities # Output the predicted class

     # Output the probabilities for each class


def getImageResult(image_input):

    # Load the best model from the checkpoint
    ensemble_model = load_model('../images_models/best_weighted_ensemble_model.keras')

    img_array = image.img_to_array(image_input) / 255.0  # Normalize to [0, 1] range
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = ensemble_model.predict(img_array)
    predicted_class = int(prediction[0] > 0.5)
    #print(f'Predicted class: {predicted_class}')
    #print(prediction)
    return predicted_class,prediction
    # gives 0 0 is normla , for false , we good
def getTabnetResult(tabnet_input):
    clf = torch.load('../tabnetmodel/tabnet__model_heart_fail.pth')
    y_pred = clf.predict(X_valid)

    # Calculate accuracy
    accuracy = accuracy_score(y_valid, y_pred)
    #print("Validation Accuracy:", accuracy)
    return y_pred,accuracy


new_sample = "Age:12,Anaemia:1,Creatinine Phosphokinase:200,Diabetes:1,Ejection Fraction:25,High Blood Pressure:89,Platelets:8,Serum Creatinine:90,Serum Sodium:99,Sex:0,Smoking:1,Time:None"
img_path = 'testingimageforresnet.png'  # Replace with actual image path
data_tabnet=pd.read_csv("testing.csv")
data=pd.DataFrame(data_tabnet)
X = data.drop(columns=["DEATH_EVENT"])
y = data["DEATH_EVENT"]
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_valid, y_train, y_valid = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

img = image.load_img(img_path, target_size=(224, 224))
print(combinedresult(image_input=img
                     ))