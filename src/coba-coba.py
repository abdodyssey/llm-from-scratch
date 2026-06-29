from pathlib import Path

text = Path('data/shakespeare.txt').read_text()

chars = sorted(set(text))
vocab_size = len(chars)

# print(vocab_size)
# # isi dataset
# print(chars[:20])

stoi = {c:i for i, c in enumerate(chars)}
itos = {i:c for i, c in enumerate(chars)}


# kenapa spasi dan \n -> termasuk unique char?
# kalau diabaikan -> hasil decode bakal rusak -> hi guys -> higuys

def encode(text: str) -> list[int]:
  return [stoi[c] for c in text]

z
def decode(tokens: int) -> str:
  return "".join([itos[i] for i in tokens])


# print(encode('To be'))

# how to test decode
sample = "To be"

tokens = encode(sample)
# print(decode(tokens))


# Bidgrams

def build_bigrams(tokens):
  result = []
  for i in range(0,len(tokens)-1):
    result.append((tokens[i], tokens[i + 1]))
  return result

tokens = [0, 1, 0, 1, 0] #ababa
bigrams = build_bigrams(tokens)
print("Bigrams: ", build_bigrams(tokens))
# sampe disin -> model belum bisa memprediksi token selanjutnya -> baru ngasih pasangan saja

print("===========================================================")
# next step
# pasangan mana yang paling sering muncul,
# probabilitas transisinya,
# kalau token sekarang 20, berikutnya paling mungkin apa.

# transition inspector


def transition_inspector(bigrams):
  result = {}
  for current, next_token in bigrams:

    if current not in result:
      result[current] = {} # -> result = {a: {}}

    if next_token not in result[current]:
      result[current][next_token] = 1
    else:
      result[current][next_token] += 1

  return result

transitions =  transition_inspector(bigrams)
print(transitions)


def predict_next(current_token, transitions):

  if current_token not in transitions:
    return None

  return max(transitions[current_token], key=transitions[current_token].get)


print(predict_next(0, transitions))

# Kalau ini -> Real GPT -> not return None -> GPT tidak memakai tabel frekuensi seperti -> Bigram 
# GPT -> menghasilkan probability distribution -> untuk seluruh vocab -> menggunakan neural network -> selalu bisa memberikan prediksi (meskipun mungkin probabilitasnya kecil)
# None -> solution -> Bigram LM yg sederhana





# ==================
import torch

tokens = torch.tensor([1,2,3,4,5,6,7,8])

block_size = 3


# ambil 2 windows
x1 = tokens[0:3]
x2 = tokens[2:5]

# kalau ingin nerima 2 window sekaligus
x = torch.stack([x1, x2])
print(x)
print(x.shape)



data = torch.tensor([1,2,3,4,5,6,7,8])
block_size = 3
batch_size = 2

def get_batch(data, block_size, batch_size):
  ix = torch.randint(len(data) - block_size, (batch_size,)) 
  
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size + 1] for i in ix])

  return x, y



x , y = get_batch(data, 3, 2)

print(x)
print(y)



# i = tensor(1) -> dianggap 1 -> Pytorch -> mendukung 0-dimensional Tensor (scalar tensor) sebagai -> index
ix = torch.randint(5, (3,))

print(i.item() for i in ix)

print(ix)