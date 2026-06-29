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