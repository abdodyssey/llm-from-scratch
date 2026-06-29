from pathlib import Path

text = Path("data/shakespeare.txt").read_text()

chars = sorted(set(text))

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

def encode(text):
    return [stoi[c] for c in text]

def decode(tokens):
    return "".join([itos[i] for i in tokens])