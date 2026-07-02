kenapa Bigram tidak cukup?
Bigram
-> hanya melihat 1 token sebelumnya.
-> hanya belajar P(next | current)

Transformer 
-> ingin semua token -> bisa "bertanya" -> di antara token yang sudah kulihat -> mana token yang paling penting untukku?

misal : I love machine learning
saat learning -> diproses -> Aku paling butuh informasi dari:
                              machine  ⭐⭐⭐⭐⭐
                              love     ⭐⭐
                              I         ⭐

artinya -> setiap token -> memberikan perhatian (attention) -> ke token -> lain

Q, K, V -> setiap token punya ini.
Q -> Query -> Pertanyaan
K -> Key -> Label yang dimiliki setiap token
V -> Value -> Isi informasi yang sebenarnya

-> memproyeksikan embedding -> ke 3 ruang yang berbeda.

Kenapa memakai linear (nn.Linear)
-> Query ingin melihat:
"Apa yang sedang dicari?"
Key ingin melihat:
"Apa identitasku?"
Value ingin melihat:
"Informasi apa yang kubawa?"
Masing-masing membutuhkan representasi yang berbeda.

Attention dimulai -> ketika -> Q dan K -> dibandingkan. -> seberapa cocok query token A dengan Key token B?

Hasil dari perbandingan -> bukan tensor lagi -> tapi -> score perhatian (attention score).

Kenapa gak embedding nya saja yang dibandingkan?
Embedding -> identitas lengkap seseorang

Case -> wawancara 
- CV
- Pertanyaan HRD
- Jawabanmu

Dari embedding yang sama -> dibuat 3 peran
1. Query -> Apa yang sedang aku cari, misal "learning" -> "Siapa yang punya informasi yg kubutuhkan"
2. Key -> Apa identitas ku?, misal "machine" -> aku punya informasi tentang machine.
3. Value -> Kalau memang kamu memilihku -> inilah informasi yang akan kuberikan.

====
Embedding
     │
 ┌───┼────┐
 │   │    │
 Q   K    V

Q
│
│ dibandingkan
▼
K

Kalau cocok

ambil V
======

Jadi secara sistematis:
Q = Wq · x
K = Wk · x
V = Wv · x

Karena bobot(Weight)nya berbeda ->  hasil proyeksi berbeda

=====
Selama ini kita punya:
Embedding
        │
        ▼
+------------------+
| Linear (Query)   | ---> Q
+------------------+

+------------------+
| Linear (Key)     | ---> K
+------------------+

+------------------+
| Linear (Value)   | ---> V
+------------------+

yang akan dilakukan transformer -> Q × Kᵀ

Attention -> matrik skor kesamaan (similiarity) antar token, bukan isi informasinya.
