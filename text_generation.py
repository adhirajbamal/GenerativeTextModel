# Generative Text Model - CODTECH Internship Task
# Implementation using both GPT-2 and LSTM approaches

## 1. Setup and Installation
!pip install transformers torch numpy

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from lstm_text_generator import LSTMModel  # Our custom LSTM implementation

## 2. GPT-2 Implementation

def load_gpt2(model_name='gpt2-medium'):
    """Load GPT-2 model and tokenizer"""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

def generate_text_gpt2(prompt, model, tokenizer, max_length=100, temperature=0.7):
    """Generate text using GPT-2"""
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
tokenizer, gpt2_model = load_gpt2()
prompt = "The future of artificial intelligence"
generated_text = generate_text_gpt2(prompt, gpt2_model, tokenizer)
print(generated_text)

## 3. LSTM Implementation

# Data preparation
class TextDataset(Dataset):
    def __init__(self, file_path, seq_length=100):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.text = file.read()
        
        self.chars = sorted(list(set(self.text)))
        self.char_to_idx = {ch:i for i,ch in enumerate(self.chars)}
        self.idx_to_char = {i:ch for i,ch in enumerate(self.chars)}
        
        self.data = [self.char_to_idx[ch] for ch in self.text]
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        seq_in = self.data[idx:idx+self.seq_length]
        seq_out = self.data[idx+1:idx+self.seq_length+1]
        return torch.tensor(seq_in), torch.tensor(seq_out)

# Training function
def train_lstm(model, dataset, epochs=10, batch_size=64):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y.view(-1))
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Text generation
def generate_text_lstm(model, dataset, start_text, length=100, temperature=1.0):
    model.eval()
    chars = [ch for ch in start_text]
    
    for _ in range(length):
        x = torch.tensor([dataset.char_to_idx[ch] for ch in chars[-dataset.seq_length:]])
        x = x.unsqueeze(0)
        
        with torch.no_grad():
            pred = model(x)
        
        pred = pred[0,-1,:] / temperature
        probs = torch.softmax(pred, dim=0)
        next_char = dataset.idx_to_char[torch.multinomial(probs, 1).item()]
        chars.append(next_char)
    
    return ''.join(chars)

# Example usage
dataset = TextDataset('data/shakespeare.txt')
lstm_model = LSTMModel(len(dataset.chars), 256, 512, 2)
train_lstm(lstm_model, dataset, epochs=5)

generated_text = generate_text_lstm(lstm_model, dataset, "ROMEO:", length=200)
print(generated_text)

## 4. Interactive Demo

def interactive_demo():
    print("CODTECH TEXT GENERATION DEMO")
    print("---------------------------")
    print("1. GPT-2 Generation")
    print("2. LSTM Generation")
    print("3. Exit")
    
    choice = input("Select option (1-3): ")
    
    if choice == '1':
        prompt = input("Enter your prompt: ")
        length = int(input("Output length (50-500): "))
        temp = float(input("Temperature (0.1-1.0): "))
        print("\nGenerating text...\n")
        print(generate_text_gpt2(prompt, gpt2_model, tokenizer, length, temp))
    elif choice == '2':
        prompt = input("Enter starting text (max 100 chars): ")[:100]
        length = int(input("Output length (50-500): "))
        temp = float(input("Creativity (0.1-2.0): "))
        print("\nGenerating text...\n")
        print(generate_text_lstm(lstm_model, dataset, prompt, length, temp))
    elif choice == '3':
        return
    else:
        print("Invalid choice")

interactive_demo()
