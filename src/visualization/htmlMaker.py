from transformers import GPT2Tokenizer, GPT2Model
import torch
from pyvis.network import Network

# Initialize tokenizer and model, set pad token
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2Model.from_pretrained('gpt2', output_attentions=True)

# Input text
text = "a love letter to my beloved, and I am so grateful for it. I am so so sorry for all the love and support that I have received from you. "
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Generate output and attention
outputs = model(**inputs, output_attentions=True)
attentions = outputs.attentions

# Choose the last layer and the first head for visualization
attention_matrix = attentions[-1][0][0].detach().numpy()

# Tokenize the text for labeling nodes, and clean the tokens
tokens = tokenizer.tokenize(text)
clean_tokens = [token.replace("Ġ", " ") for token in tokens] 

# Create a network graph
net = Network(height="750px", width="100%", bgcolor="#e8d1d1", font_color="white", notebook=True)

# Add nodes
heart_svg = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbDpzcGFjZT0icHJlc2VydmUiIHZpZXdCb3g9IjAgMCA0NSA0NSI+PGRlZnMgZmlsbD0iI2QzMjUxOCI+PGNsaXBQYXRoIGlkPSJhIiBjbGlwUGF0aFVuaXRzPSJ1c2VyU3BhY2VPblVzZSIgZmlsbD0iI2QzMjUxOCI+PHBhdGggZD0iTTAgMzZoMzZWMEgwWiIgZmlsbD0iI2QzMjUxOCIvPjwvY2xpcFBhdGg+PC9kZWZzPjxnIGZpbGw9IiM0ZjY0NmEiIGNsaXAtcGF0aD0idXJsKCNhKSIgdHJhbnNmb3JtPSJtYXRyaXgoMS4yNSAwIDAgLTEuMjUgMCA0NSkiPjxwYXRoIGQ9Ik0zNS44ODQ4IDI0LjE2NjVjMCA1LjQ1LTQuNDE4IDkuODY4LTkuODY3IDkuODY4LTMuMzA4IDAtNi4yMjctMS42MzMtOC4wMTgtNC4xMjktMS43OSAyLjQ5Ni00LjcxIDQuMTI5LTguMDE3IDQuMTI5LTUuNDUgMC05Ljg2OC00LjQxOC05Ljg2OC05Ljg2OCAwLS43NzIuMDk4LTEuNTIuMjY2LTIuMjQxIDEuMzcxLTguNTEyIDEwLjgzNS0xNy40OTQgMTcuNjE5LTE5Ljk2IDYuNzgzIDIuNDY2IDE2LjI0OSAxMS40NDggMTcuNjE3IDE5Ljk2LjE3LjcyMS4yNjggMS40NjkuMjY4IDIuMjQxIiBmaWxsPSIjZDMyNTE4Ii8+PC9nPjwvc3ZnPg=="
# Add nodes
for i, token in enumerate(clean_tokens):
    net.add_node(i, label=token, title=token, color='#630d03', shape='image', image=heart_svg)

# Add edges with attention weight as width
threshold = 0.01  
for i in range(len(clean_tokens)):
    for j in range(len(clean_tokens)):
        weight = attention_matrix[i][j]
        if weight > threshold:
            net.add_edge(i, j, value=float(weight), color='#ff5252')

net.set_options("""
var options = {
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -80000,
      "centralGravity": 0.1,
      "springLength": 75,
      "springConstant": 0.04,
      "damping": 0.09,
      "avoidOverlap": 0
    },
    "maxVelocity": 50,
    "minVelocity": 0.75,
    "solver": "barnesHut",
    "timestep": 0.5
  }
}
""")

net.show("index.html")
