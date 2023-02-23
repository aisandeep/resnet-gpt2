model.py


import torch
import torch.nn as nn
from transformers import GPT2Model

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.9.0', 'resnet101', pretrained=True)
        self.fc = nn.Linear(1000, 768)
        
    def forward(self, x):
        with torch.no_grad():
            x = self.resnet(x)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        
    def forward(self, x, html):
        x = x.unsqueeze(0) # Add batch dimension
        html = self.gpt2(html)[0]
        return html, None # Return predicted HTML and loss (set to None since we don't need to compute it during inference)






train.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms
from model import Encoder, Decoder

class TableDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.loader = default_loader
        self.samples = self._find_samples()
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, html_path = self.samples[index]
        image = self.loader(path)
        html = open(html_path, 'r').read()
        if self.transform is not None:
            image = self.transform(image)
        return image, html
    
    def _find_samples(self):
        samples = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    path = os.path.join(dirpath, filename)
                    html_path = os.path.join(dirpath, filename.replace('.jpg', '.html'))
                    samples.append((path, html_path))
        return samples

class TableToHTMLModel(nn.Module):
    def __init__(self):
        super(TableToHTMLModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x, html):
        x = self.encoder(x)
        html, _ = self.decoder(x, html)
        return html
    
def train(model, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    for batch_idx, (image, html) in enumerate(train_loader):
        optimizer.zero_grad()
        html_pred = model(image, html)
        loss = criterion(html_pred, html)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return train_loss / len(train_loader)

def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for image, html in val_loader:
            html_pred = model(image, html)
            val_loss += criterion(html_pred,
    



attention:


import torch
import torch.nn as nn
from transformers import GPT2Model

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        self.fc = nn.Linear(768, 50256)
        
    def forward(self, x, html):
        x = x.unsqueeze(0) # Add batch dimension
        html_inputs = self.gpt2.transformer.wte(html) # Embed input HTML tokens
        html_inputs = html_inputs.unsqueeze(0) # Add batch dimension
        hidden_states, _ = self.gpt2(inputs_embeds=html_inputs) # Compute initial hidden states
        hidden_states = hidden_states.transpose(0, 1) # Transpose to (seq_len, batch_size, hidden_dim)
        x = x.transpose(0, 1) # Transpose to (seq_len, batch_size, hidden_dim)
        attn_output, attn_weights = self.attention(x, hidden_states, hidden_states) # Compute cross-attention
        html_pred = self.fc(attn_output.squeeze(0)) # Generate HTML tokens from cross-attention output
        return html_pred, attn_weights



data about attention head


In the self.attention layer in the Decoder class, the hidden states are passed twice 
because this is how the nn.MultiheadAttention module works.

The nn.MultiheadAttention module takes three inputs: query, key, and value. In this case, 
the query is the current decoder hidden state, while the key and value are the entire sequence of decoder 
hidden states produced so far. By passing the entire sequence of hidden states as both the key and value, 
the nn.MultiheadAttention module is able to compute an attention distribution over the entire sequence, 
and use that distribution to compute a weighted average of the hidden states.

So in the line attn_output, attn_weights = self.attention(x, hidden_states, hidden_states), hidden_states 
is passed twice because it is used both as the key and value inputs to the attention mechanism. 
The attn_output tensor is the weighted average of the hidden states, computed using the attention 
distribution returned by the nn.MultiheadAttention module.




