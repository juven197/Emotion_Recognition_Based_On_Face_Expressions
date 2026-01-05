# Emotion Recognition Using Swin Transformer

## Deskripsi Proyek
Emotion Recognition merupakan sebuah aplikasi berbasis desktop yang dirancang untuk mendeteksi emosi manusia berdasarkan ekspresi wajah. Aplikasi ini dibangun menggunakan Streamlit dan menerapkan metode Swin Transformer sebagai model utama dalam proses klasifikasi emosi.

Program ini mampu melakukan prediksi terhadap empat jenis emosi, yaitu happy, sad, angry, dan neutral. Selain itu, sistem ini menyediakan dua mode deteksi, yaitu melalui unggahan gambar wajah (Upload Image) dan deteksi secara langsung menggunakan kamera (Real Time).

## Struktur File
Berikut adalah struktur file utama pada repository ini:
- Manual Book.pdf : Berisi panduan penggunaan aplikasi Emotion Recognition secara bertahap.
- model_best_acc.pkl : Model Swin Transformer dengan performa terbaik berdasarkan nilai akurasi.
- model_best_loss.pkl : Model Swin Transformer dengan performa terbaik berdasarkan nilai loss.
- model_fix.py : File utama untuk menjalankan aplikasi Streamlit.
- requirements.txt : Daftar library Python yang diperlukan untuk menjalankan program.

## Instalasi dan Menjalankan Program
1. Unduh repository dengan memilih menu 'Code' kemudian klik 'Download ZIP'.
2. Setelah itu, ekstrak folder ke direktori yang diinginkan.
3. Buka Command Prompt atau Terminal, kemudian arahkan ke folder project.
4. Install library dengan perintah `pip install -r requirements.txt`
5. Setelah proses instalasi selesai, jalankan program dengan perintah `streamlit run model_fix.py`
6. Browser akan otomatis terbuka dan menampilkan halaman utama aplikasi Emotion Recognition.



