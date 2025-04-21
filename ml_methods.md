
# Metode Machine Learning untuk Ekstraksi Informasi dari Invoice

Dokumen ini membahas teori singkat tentang dua pendekatan utama yang digunakan dalam sistem ekstraksi informasi invoice otomatis: Named Entity Recognition (NER) kustom dan model layout-aware (misal, LayoutLM/CUTIE).

## 1. Named Entity Recognition (NER) Kustom dengan spaCy

**Definisi**:  
NER adalah teknik NLP untuk mengidentifikasi dan mengklasifikasikan entitas penting di dalam teks, seperti nama, tanggal, nomor invoice, dsb.

**Praktik pada Invoice**:  
- Melatih model NER pada data invoice dengan label: invoice_number, invoice_date, customer_name, item_name, item_quantity, item_price, subtotal, total, tax, dsb.
- Output model berupa posisi dan label setiap entitas dalam dokumen.

**Keunggulan**:  
- Dapat beradaptasi pada format/bahasa berbeda.
- Ketika dilatih memadai, efektif untuk mendeteksi entitas utama meski format dokumen bervariasi ringan.

## 2. Model Layout-Aware (LayoutLM/CUTIE)

**Definisi**:  
Model layout-aware seperti LayoutLM atau CUTIE mengintegrasikan informasi teks (dari OCR) dan posisi spasial (koordinat bounding box) untuk memahami struktur dokumen dengan lebih baik.

**Praktik pada Invoice**:  
- Model menerima input: token OCR + bounding box posisi tiap token.
- Memungkinkan deteksi entitas dan relasi walaupun lokasi, urutan, atau bahasa berbeda-beda.
- Khusus efektif untuk memahami dokumen yang memiliki struktur tabel atau multi-kolom.

**Keunggulan**:  
- Sangat tangguh untuk layout variatif atau dokumen multi-bahasa/mata uang.
- Dapat belajar pola visual dokumen transaksi dari data pelatihan.

**Referensi**:
- [spaCy Custom NER](https://spacy.io/usage/training)
- [LayoutLM Paper (Microsoft)](https://arxiv.org/abs/1912.13318)
- [CUTIE Paper (Tencent)](https://arxiv.org/abs/1903.12363)

---

## Rangkuman Implementasi

- Setelah OCR selesai, gunakan model NER custom dan/atau layout-aware untuk mengekstrak entitas dan relasi dari hasil OCR (teks & posisi eksplisit).
- Hindari rule/regex/manual pattern-matching saat inferensi (agar mendukung multi-bahasa & layout variatif).
- Output pipeline didasarkan pada hasil model machine learning, bukan heuristik.

