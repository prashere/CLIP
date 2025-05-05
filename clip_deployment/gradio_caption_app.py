import torch
import torch.nn.functional as F
from torchvision import transforms
import gradio as gr
from model import CLIPModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CLIPModel()
model.load_state_dict(torch.load(
    "model/final_epoch_clip.pt", map_location=device))
model.to(device)
model.eval()

data = torch.load("caption/caption_embeddings.pt")
captions = data["captions"]
text_embeddings = data["embeddings"].to(device)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.481, 0.457, 0.408],
                         std=[0.268, 0.261, 0.275]),
])


def predict_caption(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = model.image_encoder(image)
        img_emb = F.normalize(img_emb, p=2, dim=-1)
        sims = img_emb @ text_embeddings.T
        topk = 2
        values, indices = sims.topk(topk, dim=-1)
        indices = indices.squeeze().cpu().tolist()
        scores = values.squeeze().cpu().tolist()

    results = [
        f"{i+1}. {captions[idx]} (score={scores[i]:.4f})" for i, idx in enumerate(indices)]
    return "\n".join(results)


interface = gr.Interface(
    fn=predict_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Caption Matching with CLIP",
    description="Upload an image and get the top 5 matching captions."
)

if __name__ == "__main__":
    interface.launch()
