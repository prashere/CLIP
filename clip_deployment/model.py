import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import DistilBertModel

class ImageEncoder(nn.Module):
  def __init__(self, embedding_dim):
    super(ImageEncoder, self).__init__()

    resnet50 = models.resnet50(pretrained = True)
    modules = list(resnet50.children()) [:-1] # Not the last classification layer

    self.model = nn.Sequential(*modules)
    self.fc1 = nn.Linear(resnet50.fc.in_features, 512)
    self.img_projection = nn.Linear(512, embedding_dim)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.model(x).squeeze()
    x = self.relu(self.fc1(x))
    embeddings = self.img_projection(x)
    return embeddings
  

class TextEncoder(nn.Module):
  def __init__(self, embedding_dim):
    super(TextEncoder,self).__init__()
    self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
    self.text_projection = nn.Linear(self.bert.config.hidden_size, embedding_dim)

  def forward(self, input_ids, attention_mask):
    out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    out = out.last_hidden_state[:, 0]  # CLS token
    embeddings = self.text_projection(out)
    return embeddings
  
class CLIPModel(nn.Module):
    def __init__(self, embed_dim=256):
        super(CLIPModel, self).__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)

    def forward(self, input_ids, attention_mask, images):
        text_embeddings = self.text_encoder(input_ids, attention_mask)
        image_embeddings = self.image_encoder(images)

        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1) ####

        return text_embeddings, image_embeddings

    def encode_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = out.last_hidden_state[:, 0]  # CLS token
        return self.text_proj(pooled_output)

    def encode_image(self, images):
        features = self.image_encoder(images).squeeze()
        return self.image_proj(features)