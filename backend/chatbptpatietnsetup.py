from datasets import load_dataset

dataset = load_dataset("not-lain/wikipedia")
from sentence_transformers import SentenceTransformer
ST = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
def embed(batch):

    information = batch["text"]
    return {"embeddings" : ST.encode(information)}

dataset = dataset.map(embed,batched=True,batch_size=16)
dataset.push_to_hub("not-lain/wikipedia", revision="embedded")
