from sentence_transformers import SentenceTransformer
from datasets import load_dataset

ST = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

dataset = load_dataset("not-lain/wikipedia",revision = "embedded")

data = dataset["train"]
data = data.add_faiss_index("embeddings") # column name that has the embeddings of the dataset

def search(query: str, k: int = 3 ):
    """a function that embeds a new query and returns the most probable results"""
    embedded_query = ST.encode(query) # embed new query
    scores, retrieved_examples = data.get_nearest_examples( # retrieve results
        "embeddings", embedded_query, # compare our new embedded query with the dataset embeddings
        k=k # get only top k results
    )
    return scores, retrieved_examples
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config
)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
SYS_PROMPT = """You are an assistant for answering questions.
You answer medical questions related to the patient while referring to the pateint RAG file , do not offer dangerous recommendations, 
"""



def format_prompt(prompt, retrieved_documents, k, patient_info):
    PROMPT = f"Question: {prompt}\n"

    # Add patient info context if available
    if patient_info:
        PROMPT += f"Patient Info: Name: {patient_info['name']}, Age: {patient_info['age']}, Medical History: {', '.join(patient_info['medical_history'])}\n"

    PROMPT += "Context:\n"
    for idx in range(k):
        PROMPT += f"{retrieved_documents['text'][idx]}\n"

    return PROMPT


def generate(formatted_prompt):
  formatted_prompt = formatted_prompt[:2000] # to avoid GPU OOM
  messages = [{"role":"system","content":SYS_PROMPT},{"role":"user","content":formatted_prompt}]

  input_ids = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      return_tensors="pt"
  ).to(model.device)
  outputs = model.generate(
      input_ids,
      max_new_tokens=1024,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )
  response = outputs[0][input_ids.shape[-1]:]
  return tokenizer.decode(response, skip_special_tokens=True)
import json


def load_patient_info(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)
def get_patient_info(patient_id):
    return patient_info if patient_info['patient_id'] == patient_id else None

patient_info = load_patient_info("patient_info.json")  # Path to your JSON file



def rag_chatbot(prompt: str, patient_id: int, k: int = 2):
    patient_info = get_patient_info(patient_id)
    scores, retrieved_documents = search(prompt, k)
    formatted_prompt = format_prompt(prompt, retrieved_documents, k, patient_info)
    return generate(formatted_prompt)
