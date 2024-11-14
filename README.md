TF-IDF vs AI Gemini Text Summarization Comparison
Deskripsi
Proyek ini membandingkan dua metode peringkasan teks: TF-IDF (Term Frequency-Inverse Document Frequency) dan AI Gemini untuk menganalisis kualitas peringkasan teks dalam bahasa Indonesia. Aplikasi ini dirancang untuk membantu memahami seberapa efektif kedua metode ini dalam merangkum artikel atau teks panjang.

Fitur
Peringkasan dengan TF-IDF: Menggunakan algoritma TF-IDF untuk merangkum teks berdasarkan frekuensi kata.
Peringkasan dengan AI Gemini: Menggunakan model AI Gemini untuk menghasilkan ringkasan otomatis.
Perbandingan Kualitas: Menggunakan metrik ROUGE untuk membandingkan kualitas ringkasan yang dihasilkan oleh kedua metode.
Analisis Perbandingan: Menampilkan hasil perbandingan antara TF-IDF dan AI Gemini dalam hal presisi, recall, dan F1-score.
Teknologi yang Digunakan
Backend: Django
Summarization: TF-IDF, AI Gemini
Evaluasi Kualitas: ROUGE
Library Python: sklearn, nltk, gemini, rouge_score
Database: (sebutkan jenis database yang digunakan, misalnya SQLite atau MySQL)
Instalasi
Clone repository ini:

bash
Salin kode
git clone https://github.com/username/nama-repository.git
Masuk ke direktori proyek:

bash
Salin kode
cd nama-repository
Buat dan aktifkan virtual environment (opsional tapi direkomendasikan):

bash
Salin kode
python -m venv env
source env/bin/activate  # Untuk MacOS/Linux
env\Scripts\activate     # Untuk Windows
Install dependensi proyek:

bash
Salin kode
pip install -r requirements.txt
Jalankan migrasi database:

bash
Salin kode
python manage.py migrate
Jalankan server lokal:

bash
Salin kode
python manage.py runserver
Akses aplikasi di browser Anda di http://127.0.0.1:8000.

Cara Menggunakan
Unggah teks atau artikel yang ingin diringkas.
Pilih metode peringkasan yang diinginkan (TF-IDF atau AI Gemini).
Aplikasi akan menghasilkan ringkasan dan membandingkan kualitas keduanya menggunakan metrik ROUGE.
Evaluasi Kualitas
Presisi, Recall, F1-score: Metrik ini digunakan untuk mengevaluasi hasil peringkasan dari kedua metode dan memberikan wawasan tentang seberapa baik kedua teknik ini bekerja dalam merangkum teks.
Kontribusi
Kontribusi sangat diterima! Silakan ajukan issue atau buat pull request jika Anda memiliki saran perbaikan atau fitur baru untuk proyek ini.
