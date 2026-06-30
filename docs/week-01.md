# Week 1 — Tokenization & Bigram Baseline

## Tujuan

Membangun fondasi Language Model paling sederhana sebelum menggunakan PyTorch dan Transformer.

Target minggu ini:

- Character Vocabulary
- stoi & itos
- encode()
- decode()
- Bigram Generator
- Bigram Transition Inspector
- Top-10 Next Character Probabilities
- Sample Text Generation

---

# 1. Character Vocabulary

## Tujuan

Mengambil seluruh karakter unik dari dataset Shakespeare.

```python
chars = sorted(set(text))
```

Contoh:

```text
['\n', ' ', '!', "'", ',', '.', ':', ';', '?', 'A', ...]
```

Jumlah vocabulary:

```python
vocab_size = len(chars)
```

Pada dataset Shakespeare:

```text
65 karakter unik
```

---

# 2. stoi (String To Integer)

Membuat mapping karakter menjadi ID.

```python
stoi = {
    'a':24,
    'b':25,
    ...
}
```

Tujuan:

Model Machine Learning tidak memahami karakter.

Model hanya memahami angka.

---

# 3. itos (Integer To String)

Kebalikan dari stoi.

```python
itos = {
24:'a',
25:'b'
}
```

Digunakan saat mengubah output model menjadi teks kembali.

---

# 4. encode()

Mengubah string menjadi token ID.

Input:

```text
Hello
```

Output:

```text
[20,43,50,50,53]
```

---

# 5. decode()

Mengubah token kembali menjadi string.

Input:

```text
[20,43,50,50,53]
```

Output:

```text
Hello
```

---

# 6. Bigram Generator

## Tujuan

Menghasilkan pasangan dua karakter berurutan.

Misal:

```text
Hello
```

Menjadi:

```text
(H,e)

(e,l)

(l,l)

(l,o)
```

Implementasi:

```python
for i in range(len(text)-1):
    bigrams.append((text[i], text[i+1]))
```

Output:

```text
Total bigrams:
1115393
```

---

# 7. Transition Table

Menghitung berapa kali karakter berikutnya muncul.

Contoh:

```text
'a'
```

Menghasilkan:

```text
{
'n':10197,
't':8339,
'l':4149,
...
}
```

Belum berupa probabilitas.

Masih berupa jumlah kemunculan.

---

# 8. Probability Table

Mengubah jumlah kemunculan menjadi probabilitas.

Rumus:

```
P(next|current)
=
count(current,next)
/ total(current)
```

Contoh:

```text
'a'

↓

'n' = 0.18

't' = 0.15

'l' = 0.08
```

Total seluruh probabilitas:

```text
1.0
```

---

# 9. Top-10 Next Character

Menampilkan 10 karakter dengan probabilitas terbesar.

Contoh:

```text
Character: '\n'

'\n' -> 0.1806

'T' -> 0.1036

'A' -> 0.0958

...
```

Tujuan:

Melihat bagaimana model statistik memahami dataset.

---

# 10. Random Text Generation

Menghasilkan teks menggunakan distribusi probabilitas bigram.

Implementasi menggunakan:

```python
random.choices()
```

Output:

```text
TERureaind anes d my VI...
```

Walaupun belum membentuk kalimat sempurna, pola karakter mulai menyerupai bahasa Inggris.

---

# Struktur File

```
src/

tokenizer.py

bigram.py
```

---

# Yang Dipelajari

- Character-level tokenization
- Vocabulary
- Integer Encoding
- Integer Decoding
- Bigram
- Conditional Probability
- Probabilistic Text Generation

---

# Keterbatasan Bigram

Model hanya melihat:

```
1 karakter

↓

menebak

↓

1 karakter berikutnya
```

Contoh:

```
a

↓

?
```

Model tidak mengetahui konteks sebelumnya.

Misalnya:

```
The ki

↓

?

```

Model tidak tahu bahwa kata berikutnya kemungkinan adalah:

```
king
```

Karena ia hanya melihat satu karakter terakhir.

---

# Alasan Masuk ke Week 2

Bigram menggunakan probabilitas hasil perhitungan statistik.

Agar model dapat belajar sendiri dari data menggunakan optimisasi (gradient descent), kita beralih ke Neural Bigram dengan PyTorch.




===================
Improve yang kita lakukan (dibanding implementasi paling sederhana)

Ini penting untuk dicatat juga.

1. Refactor menjadi fungsi

Sebelumnya semuanya berada di global scope.

Diubah menjadi:

build_bigrams()
build_transitions()
build_probabilities()
generate_text()
main()

Alasan:

lebih modular,
mudah diuji,
mudah dikembangkan.
2. Memisahkan tokenizer

Membuat:

tokenizer.py

berisi:

encode()
decode()
stoi
itos

Alasan:

tidak duplikasi kode,
akan dipakai lagi di train.py, generate.py, dan model.py.
3. Menambahkan Transition Inspector

Ini sebenarnya bukan syarat mutlak language model, tetapi diminta oleh pembimbing.

Alasan:

membantu interpretasi data,
memverifikasi distribusi karakter.
4. Menambahkan Top-10 Probability

Daripada mencetak seluruh distribusi (puluhan karakter), kita hanya mengambil:

sorted(...)[0:10]

Alasan:

lebih mudah dianalisis,
sesuai kebutuhan submit.
5. Membuat Text Generator

Menggunakan:

random.choices(weights=...)

bukan memilih probabilitas terbesar.

Alasan:
Kalau selalu memilih probabilitas terbesar (greedy decoding), hasilnya menjadi sangat repetitif. Dengan sampling berbobot, teks yang dihasilkan lebih bervariasi meskipun tetap mengikuti distribusi probabilitas.