# caption_encoder.py
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer
import json
from model import CLIPModel  # make sure your model code is in model.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load captions
with open("caption/captions_1000.json", "r") as f:
    captions = json.load(f)["captions"]

# Tokenize
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
inputs = tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Load model
model = CLIPModel()
model.load_state_dict(torch.load("model/final_epoch_clip.pt", map_location=device))
model.to(device)
model.eval()

# Encode and normalize
with torch.no_grad():
    text_embeddings = model.text_encoder(input_ids, attention_mask)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

# Save to file
torch.save({
    "captions": captions,
    "embeddings": text_embeddings.cpu()
}, "caption/caption_embeddings.pt")

print("Caption embeddings saved to caption_embeddings.pt")
