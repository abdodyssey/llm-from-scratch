# Week 2 — PyTorch Training Basics

## Tujuan

Mengubah Bigram berbasis statistik menjadi Bigram Neural Network menggunakan PyTorch.

Target minggu ini:

- Train / Validation Split
- get_batch()
- Simple Neural Language Model
- Cross Entropy Loss
- Training Loop
- Shape Assertions

---

# 1. Dataset Preparation

Dataset Shakespeare dibaca kemudian diubah menjadi token ID menggunakan tokenizer yang dibuat pada Week 1.

```python
text = Path("data/shakespeare.txt").read_text()
data = torch.tensor(encode(text), dtype=torch.long)
```

Output:

```text
Dataset size:
1115394 token
```

Semua karakter sekarang direpresentasikan sebagai integer sehingga dapat diproses oleh PyTorch.

---

# 2. Train / Validation Split

Dataset dibagi menjadi:

- 90% Training
- 10% Validation

Implementasi:

```python
n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]
```

Output:

```text
Train size
1003854

Validation size
111540
```

## Tujuan

Training digunakan untuk memperbarui parameter model.

Validation digunakan untuk mengukur kemampuan generalisasi model terhadap data yang tidak pernah dilihat.

---

# 3. get_batch()

## Tujuan

Mengambil beberapa sequence secara acak dari dataset.

Implementasi:

```python
ix = torch.randint(
    len(data)-block_size,
    (batch_size,)
)
```

Kemudian:

```python
x = torch.stack(...)
y = torch.stack(...)
```

Misalnya:

Dataset

```text
ABCDE...
```

Dengan

```text
block_size = 4
```

Maka:

```
Input (x)

A B C D

Target (y)

B C D E
```

Target selalu bergeser satu token.

---

# 4. Neural Bigram Model

Pada Week 1, prediksi dilakukan menggunakan tabel probabilitas.

Pada Week 2 diganti menggunakan:

```python
nn.Embedding()
```

Implementasi:

```python
self.token_embedding_table =
nn.Embedding(
    vocab_size,
    vocab_size
)
```

Embedding menghasilkan logits untuk setiap token.

Output shape:

```text
(B,T,C)
```

Contoh:

```
32
batch

×

8
token

×

65
vocabulary
```

Menjadi:

```
(32,8,65)
```

---

# 5. Forward Pass

Forward bertugas menghasilkan prediksi model.

Implementasi:

```python
logits =
self.token_embedding_table(idx)
```

Kemudian dilakukan reshape:

```python
logits

(B,T,C)

↓

(B*T,C)
```

Karena CrossEntropy membutuhkan bentuk:

```
(N,C)
```

Target juga diubah:

```
(B,T)

↓

(B*T)
```

---

# 6. Cross Entropy Loss

Loss digunakan untuk mengukur seberapa jauh prediksi model dari target sebenarnya.

Implementasi:

```python
loss =
F.cross_entropy(
    logits,
    targets
)
```

Semakin kecil loss berarti prediksi model semakin baik.

---

# 7. Optimizer

Optimizer yang digunakan:

```python
AdamW
```

Implementasi:

```python
optimizer =
torch.optim.AdamW(
    model.parameters(),
    lr=1e-2
)
```

Optimizer bertugas memperbarui parameter embedding berdasarkan hasil backpropagation.

---

# 8. Training Loop

Training dilakukan sebanyak:

```text
2000 step
```

Setiap iterasi:

```
Ambil batch

↓

Forward

↓

Hitung Loss

↓

Zero Grad

↓

Backward

↓

Optimizer Step
```

Implementasi:

```python
optimizer.zero_grad()

loss.backward()

optimizer.step()
```

---

# 9. Validation Loss

Agar tidak hanya melihat training loss, dibuat fungsi:

```python
estimate_loss()
```

Fungsi ini:

- mematikan gradient
- menjalankan evaluasi pada train
- menjalankan evaluasi pada validation

Menggunakan:

```python
@torch.no_grad()
```

Output:

```text
Step 0

Train
4.64

Validation
4.61

...

Step 1800

Train
2.46

Validation
2.49
```

Selisih train dan validation kecil sehingga model belum mengalami overfitting.

---

# 10. Shape Assertions

Ditambahkan assertion untuk memastikan bentuk tensor sesuai.

```python
assert x.shape == (
batch_size,
block_size
)
```

```python
assert y.shape == (
batch_size,
block_size
)
```

```python
assert logits.shape == (
batch_size * block_size,
vocab_size
)
```

```python
assert loss.ndim == 0
```

Output:

```text
Shape assertions passed.
```

---

# Struktur File

```
src/

tokenizer.py

model.py

train.py
```

---

# Yang Dipelajari

- Tensor
- Embedding
- Batch Training
- Train Validation Split
- Forward Pass
- Cross Entropy
- Backpropagation
- Optimizer
- Gradient Descent
- Shape Tracking

---

# Keterbatasan Model

Model masih merupakan Bigram Neural Network.

Setiap token diprediksi hanya berdasarkan satu token sebelumnya.

Belum terdapat:

- Positional Embedding
- Self Attention
- Multi Head Attention
- Feed Forward Network
- Transformer Block

Model juga belum mampu memahami konteks panjang.

---

# Alasan Masuk ke Week 3

Walaupun model sudah dapat belajar menggunakan gradient descent, model belum memahami hubungan antar token dalam sebuah kalimat.

Pada Week 3 akan dibangun arsitektur Transformer yang memperkenalkan:

- Positional Embedding
- Self Attention
- Multi Head Attention
- Residual Connection
- Layer Normalization

Sehingga model dapat memanfaatkan konteks yang lebih panjang dibanding Bigram.


=====================
Improvement yang kita lakukan selama Week 2

Ini menurutku penting dicatat karena bukan sekadar "mengikuti tutorial", tetapi ada beberapa perbaikan pada struktur kode.

1. Memisahkan model.py

Sebelumnya:

Semua kode (model, training, dataset) berada di train.py.

Diubah menjadi:

src/
├── model.py
├── tokenizer.py
└── train.py

Alasan:

Separation of concerns.
train.py fokus pada proses pelatihan.
model.py fokus pada arsitektur model.
Memudahkan pengembangan saat Week 3 (Transformer) dan Week 5 (generate.py).
2. Menambahkan estimate_loss()

Sebelumnya:

Hanya mencetak training loss.

Diubah menjadi:

Menghitung rata-rata training loss dan validation loss menggunakan beberapa batch.

Alasan:

Monitoring performa model lebih stabil.
Dapat mendeteksi overfitting lebih awal.
Sesuai praktik umum pada training model deep learning.
3. Menggunakan @torch.no_grad()

Sebelumnya:

Evaluasi berpotensi tetap melacak gradient.

Diubah menjadi:

@torch.no_grad()

Alasan:

Menghemat memori.
Mempercepat evaluasi.
Sesuai best practice PyTorch.
4. Menambahkan Shape Assertions

Sebelumnya:

Shape tensor hanya diasumsikan benar.

Diubah menjadi:

assert ...

Alasan:

Memverifikasi implementasi sejak awal.
Mempermudah debugging jika terjadi perubahan pada arsitektur.
Memenuhi salah satu deliverable Week 2.
5. Refactor Hyperparameter

Hyperparameter seperti:

block_size
batch_size
lr

didefinisikan di satu tempat dalam main().

Alasan:

Lebih mudah melakukan eksperimen.
Menghindari angka "magic number" yang tersebar di kode.
Menjadi dasar untuk konfigurasi yang lebih fleksibel pada minggu-minggu berikutnya.