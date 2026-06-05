import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from streamlit_sortables import sort_items

#  1. FUNGSI PRE-PROCESSING 
def preprocess_data(df):
    kolom_target = ["Engine Capacity/cc", "Horsepower", "Total Speed", "Performance", "Cars Prices", "Seats", "Torque"]
    df_clean = df.copy()

    #variabel untuk tracking
    baris_awal = len(df_clean)
    sel_diubah = 0

    def clean_number(text):
        if pd.isna(text): return np.nan
        text = str(text).lower()
        text = text.split('-')[0].split('–')[0] 
        text = re.sub(r'[^\d.,]', '', text)     
        
        # Deteksi jika data menggunakan format angka Indonesia
        if text.count('.') > 1 or (',' in text and text.rfind(',') > text.rfind('.')):
            text = text.replace('.', '')  # Hapus titik ribuan
            text = text.replace(',', '.') # Ubah koma desimal jadi titik
        else:
            text = text.replace(',', '')  # Hapus koma ribuan (format standar)
            
        try:
            return float(text)
        except:
            return np.nan
    # Pembersihan untuk setiap kolom yang ditargetkan
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].apply(clean_number)

    df_clean = df_clean.dropna()
    
    baris_dihapus = baris_awal - len(df_clean)

    #Log untuk debugging
    print("\n" + "═"*50)
    print("LAPORAN PRE-PROCESSING DATA")
    print(f"▶ Total baris data awal      : {baris_awal} baris")
    print(f"▶ Total sel yang diformat    : {sel_diubah} sel (teks satuan/rentang dihapus)")
    print(f"▶ Baris dihapus (NaN/Kosong) : {baris_dihapus} baris")
    print(f"▶ Total baris siap dihitung  : {len(df_clean)} baris")
    print("Kriteria yang dipilih:")
    for i, kriteria in enumerate(kolom_target, 1):
        print(f"{i}. {kriteria}")
    print("═"*50 + "\n")

    return df_clean

#  2. FUNGSI PEMBOBOTAN ROC 
def hitung_roc(urutan_kriteria):
    """
    Menghitung bobot ROC berdasarkan urutan prioritas.
    Rumus: Wj = (1/m) * sum(1/k) untuk k dari i sampai m
    """
    m = len(urutan_kriteria)
    bobot_roc = {}
    for i, kriteria in enumerate(urutan_kriteria):
        # Hitung sigma (penjumlahan pecahan)
        sigma_val = sum([1 / (k + 1) for k in range(i, m)])
        bobot_roc[kriteria] = sigma_val / m
        
    return bobot_roc

#  3. FUNGSI PERHITUNGAN EDAS 
def hitung_edas(df_bersih, bobot):
    tipe = {
        "Engine Capacity/cc": "benefit", "Horsepower": "benefit", "Total Speed": "benefit",
        "Seats": "benefit", "Torque": "benefit", "Performance": "cost", "Cars Prices": "cost"
    }
    
    X = df_bersih.values
    AV = np.mean(X, axis=0)
    
    rows, cols = X.shape
    PDA = np.zeros((rows, cols))
    NDA = np.zeros((rows, cols))
    
    for j, col_name in enumerate(df_bersih.columns):
        t = tipe.get(col_name, "benefit")
        for i in range(rows):
            if t == "benefit":
                PDA[i, j] = max(0, (X[i, j] - AV[j]) / AV[j])
                NDA[i, j] = max(0, (AV[j] - X[i, j]) / AV[j])
            else: # cost
                PDA[i, j] = max(0, (AV[j] - X[i, j]) / AV[j])
                NDA[i, j] = max(0, (X[i, j] - AV[j]) / AV[j])
                
    w_array = np.array([bobot.get(c, 0) for c in df_bersih.columns])
    SP = np.sum(PDA * w_array, axis=1)
    SN = np.sum(NDA * w_array, axis=1)
    
    NSP = SP / np.max(SP)
    NSN = 1 - (SN / np.max(SN))
    
    AS = (NSP + NSN) / 2
    return AS

#  KONFIGURASI DASAR 
st.set_page_config(page_title="SPK Pemilihan Mobil", layout="wide")

#  FUNGSI PENYERAGAM NAMA KOLOM
def standarisasi_kolom(df):
    """Mendeteksi kata kunci pada kolom CSV dan menyeragamkannya untuk sistem."""
    for col in df.columns:
        col_lower = col.lower()
        if 'price' in col_lower: 
            df.rename(columns={col: 'Cars Prices'}, inplace=True)
        elif 'speed' in col_lower: 
            df.rename(columns={col: 'Total Speed'}, inplace=True)
        elif 'capacity' in col_lower or 'cc' in col_lower: 
            df.rename(columns={col: 'Engine Capacity'}, inplace=True)
        elif 'horse' in col_lower or 'hp' in col_lower: 
            df.rename(columns={col: 'Horsepower'}, inplace=True)
        elif 'performance' in col_lower or 'sec' in col_lower: 
            df.rename(columns={col: 'Performance'}, inplace=True)
        elif 'seat' in col_lower: 
            df.rename(columns={col: 'Seats'}, inplace=True)
        elif 'torque' in col_lower or 'nm' in col_lower: 
            df.rename(columns={col: 'Torque'}, inplace=True)
    return df

# FUNGSI LOAD DATA DEFAULT
@st.cache_data
def load_default_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "Data_mobil_konvensional.csv")
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep=';') 
        df.columns = df.columns.str.strip()
        # PANGGIL FUNGSI DI SINI
        df = standarisasi_kolom(df) 
        return df
    return None

# INISIALISASI STATE (PENYIMPAN DATA)
if 'page' not in st.session_state:
    st.session_state.page = 'landing'  # Halaman awal
if 'df_mentah' not in st.session_state:
    st.session_state.df_mentah = load_default_data() # Otomatis load data default
if 'bobot_kriteria' not in st.session_state:
    st.session_state.bobot_kriteria = {}
if 'is_custom_data' not in st.session_state:
    st.session_state.is_custom_data = False # Penanda apakah user pakai data default atau upload sendiri

# Fungsi untuk pindah halaman
def move_to(page_name):
    st.session_state.page = page_name
    st.rerun()

with st.sidebar:
    st.subheader("Status dataset saat ini:")
    if st.session_state.is_custom_data:
        st.success("Dataset: Kustom (Diunggah User)")
    else:
        st.warning("Dataset: Default (Bawaan Sistem)")
    
    # Ditambahkan pengecekan agar tidak error jika data kosong
    if st.session_state.df_mentah is not None:
        st.write(f"Total Mobil: **{len(st.session_state.df_mentah)}** unit")
    else:
        st.write("Total Mobil: **0** unit (Data belum siap)")

# Halaman landing page
if st.session_state.page == 'landing':
    st.markdown("<h1 style='text-align: center;'>Sistem Pendukung Keputusan Pemilihan Mobil</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Menggunakan Metode EDAS & Pembobotan ROC</p>", unsafe_allow_html=True)
    st.divider()

    col_l1, col_l2 = st.columns([2, 1])
    with col_l1:
        st.write("Selamat datang! Sistem ini dirancang untuk membantu Anda" \
        " menemukan mobil terbaik berdasarkan kriteria teknis secara objektif.")
        st.write("Sistem menggunakan data default jika Anda tidak mengunggah data sendiri.")
    
    with col_l2:
        if st.button("Mulai & Upload Data", use_container_width=True, type="primary"):
            move_to('upload_page')

    st.subheader("Pilih Metode Pembobotan")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Gunakan Pembobotan ROC", use_container_width=True):
            move_to('roc_ranking')
    with c2:
        if st.button("Gunakan Pembobotan Manual", use_container_width=True):
            move_to('manual_input')

# Halaman upload data mobil dan ketentuan data
elif st.session_state.page == 'upload_page':
    st.title("Upload Data & Panduan Data Mobil")
    
    # Bagian Ketentuan (Detail sesuai Bab 3 Proposal)
    st.subheader("Ketentuan dan Format Data")
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.info("""
        **Format & Tipe Kendaraan:**
        * **Format File:** Wajib berformat **.csv**.
        * **Tipe Bahan Bakar:** Hanya menerima **Bensin (Petrol)** dan **Diesel**.
        * **Eksklusi:** Data Hybrid dan Listrik otomatis diabaikan karena perbedaan satuan teknis.
        """)
        
    with col_info2:
        st.info("""
        **Aturan Pembersihan Otomatis:**
        * **Satuan:** Teks seperti 'hp', 'cc', '$' akan dihapus otomatis.
        * **Data Rentang:** Jika tertulis '70-85 hp', sistem mengambil angka **terkecil (70)**.
        * **Data Kosong:** Baris yang memiliki kolom kosong akan dihapus untuk akurasi rata-rata.
        """)

    with st.expander("Klik untuk melihat detail 7 kriteria yang dibutuhkan"):
        st.markdown("""
        Berikut kriteria yang harus ada di file Anda:
        1. **Engine Capacity**: Kapasitas mesin dalam CC (Tipe: *Benefit*).
        2. **Horsepower**: Daya kuda/tenaga mesin (Tipe: *Benefit*).
        3. **Total Speed**: Kecepatan maksimum km/jam (Tipe: *Benefit*).
        4. **Performance**: Waktu akselerasi 0-100 km/jam dalam detik (Tipe: *Cost*).
        5. **Cars Prices**: Harga mobil dalam USD (Tipe: *Cost*).
        6. **Seats**: Kapasitas tempat duduk (Tipe: *Benefit*).
        7. **Torque**: Torsi atau daya putar mesin (Tipe: *Benefit*).
        """)
        
# Fitur Upload
    st.subheader("Upload Data Anda")
    if not st.session_state.is_custom_data:
        st.warning("ℹ️ Saat ini sistem menggunakan **Data Default** (Data_mobil_konvensional.csv).")
    else:
        st.success("✅ Sistem menggunakan **Data Kustom** yang Anda unggah.")

    # Membagi area menjadi 2 kolom (kiri lebih lebar dari kanan)
    col_upload, col_download = st.columns([3, 1])
    
    with col_upload:
        file_upload = st.file_uploader("Pilih file dataset mobil (CSV)", type=['csv'], key="uploader_main")
        
    with col_download:
        # Menambahkan sedikit ruang kosong di atas tombol 
        # agar posisi tombol sejajar ke tengah dengan kotak uploader di sebelah kiri
        st.write("") 
        st.write("") 
        
        csv_template = "company names;cars names;engines;engine capacity;horsepower;total speed;performance;cars prices;fuel types;seats;torque\n"
        st.download_button(
            label="📥 Download Format CSV",
            data=csv_template,
            file_name="Format_data_mobil.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.caption("Gunakan format ini jika ingin membuat dataset sendiri.")

    # Logika jika file berhasil diupload
    if file_upload:
        try:
            # Membaca file CSV
            df = pd.read_csv(file_upload)
            if len(df.columns) == 1: 
                file_upload.seek(0)
                df = pd.read_csv(file_upload, sep=';')
                
            df.columns = df.columns.str.strip()
            df = standarisasi_kolom(df) 
            
            # 1. Definisikan kolom/kriteria wajib yang harus ada
            kriteria_wajib = [
                "Engine Capacity", "Horsepower", "Total Speed", 
                "Performance", "Cars Prices", "Seats", "Torque"
            ]
            
            # 2. Cek apakah ada kolom wajib yang tidak ditemukan di file upload
            kolom_kurang = [col for col in kriteria_wajib if col not in df.columns]
            
            # 3. Validasi Kondisi
            if len(kolom_kurang) > 0:
                # Jika ada kolom yang kurang, tampilkan error dan JANGAN simpan data
                st.error(f"⚠️ Format data tidak sesuai! Sistem tidak menemukan kriteria berikut di file Anda: **{', '.join(kolom_kurang)}**")
                st.info("💡 Tips: Pastikan nama kolom di file Anda sesuai, atau silakan unduh 'Format CSV' untuk melihat contoh struktur yang benar.")
            else:
                # Jika semua kolom lengkap, simpan data ke sistem
                st.session_state.df_mentah = df
                st.session_state.is_custom_data = True
                st.success("✅ Data berhasil diunggah dan format sesuai! Anda sekarang bisa kembali ke Menu Utama untuk memulai perhitungan.")
                
        except Exception as e:
            # Menangkap error jika file corrupt atau bukan CSV murni
            st.error("⚠️ Terjadi kesalahan saat membaca file. Pastikan file yang diunggah benar-benar berformat teks CSV.")

            
    # Navigasi Tombol
    st.write("")
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        # PERBAIKAN: Pastikan targetnya 'landing' dan gunakan key unik
        if st.button("⬅️ Kembali ke Menu Utama", key="btn_back_to_landing"):
            move_to('landing')
    with col_nav2:
        if st.button("Lihat Detail Data Terpasang", key="btn_view_detail"):
            st.session_state.return_to = 'upload_page' # Simpan tujuan kembali
            move_to('detail_mobil')
    

#  2. Halaman pembobotan kriteria menggunakan ROC 
elif st.session_state.page == 'roc_ranking':
    st.title("Halaman Ranking Prioritas (ROC)")
    st.info("💡 Pindahkan item di sebelah kanan dengan cara klik, tahan, lalu geser (Drag and Drop) ke atas atau ke bawah. Kriteria di urutan paling atas adalah prioritas utama.")
    
    # Fitur Tombol Keterangan Benefit & Cost menggunakan Expander
    with st.expander("ℹ️ **Klik di sini untuk melihat Keterangan Benefit & Cost**"):
        st.markdown("""
        Berikut penjelasan tentang kriteria benefit dan cost:
        * **Kriteria Benefit (Keuntungan)**: Kriteria yang semakin **besar** nilainya, semakin **baik** untuk pengguna. 
          *Contoh: Engine Capacity, Horsepower, Total Speed, Seats, Torque.*
        * **Kriteria Cost (Biaya/Pengorbanan)**: Kriteria yang semakin **kecil** nilainya, semakin **baik** untuk pengguna.
          *Contoh: Cars Prices (Harga yang murah lebih baik), Performance (Waktu akselerasi yang lebih singkat lebih baik).*
        """)
    
    st.write("") # Spasi kosong

    kriteria_list = [
        "Cars Prices", "Total Speed", "Engine Capacity", 
        "Horsepower", "Performance", "Seats", "Torque"
    ]
    
    # KAMUS DATA: Memetakan kriteria dengan jenisnya
    tipe_kriteria = {
        "Cars Prices": "Cost", 
        "Total Speed": "Benefit", 
        "Engine Capacity": "Benefit", 
        "Horsepower": "Benefit", 
        "Performance": "Cost", 
        "Seats": "Benefit", 
        "Torque": "Benefit"
    }
    
    # Membagi layar menjadi 3 kolom: [Nama Kriteria] [Jenis] [Drag & Drop]
    col_kiri, col_tengah, col_kanan = st.columns([2, 1, 2])
    
    # 1. Eksekusi kolom KANAN duluan untuk mendapatkan "urutan" real-time
    with col_kanan:
        st.markdown("**↕️ Area Drag & Drop:**")
        urutan = sort_items(kriteria_list, direction='vertical' )

    # 2. Render kolom KIRI (Urutan & Nama)
    with col_kiri:
        st.markdown("**Urutan Prioritas Saat Ini:**")
        for i, val in enumerate(urutan):
            st.markdown(
                f"""
                <div style='
                    border: 1px solid #000000; 
                    border-radius: 0.25rem; 
                    margin-bottom: 0.5rem; 
                    padding: 0.3rem 0.8rem; 
                    font-size: 14px; 
                    display: flex;
                    align-items: center;
                    box-sizing: border-box;
                    box-shadow: 0px 1px 2px rgba(0,0,0,0.05); 
                '>
                    <strong style='color: #ff4b4b;'>Prioritas {i+1}</strong> &nbsp;|&nbsp; {val}
                </div>
                """, 
                unsafe_allow_html=True
            )
            
    # 3. Render kolom TENGAH (Jenis Kriteria)
    with col_tengah:
        st.markdown("**Jenis Kriteria:**")
        for val in urutan:
            jenis = tipe_kriteria[val] # Ambil jenis kriteria dari kamus data
            
            # Beri warna merah jika Cost, hijau jika Benefit
            warna_teks = "#ff4b4b" if jenis == "Cost" else "#3ec673"
            
            st.markdown(
                f"""
                <div style='
                    color: {warna_teks}; 
                    border: 1px solid #000000; 
                    border-radius: 0.25rem; 
                    margin-bottom: 0.5rem; 
                    padding: 0.3rem 0.8rem; 
                    font-size: 14px; 
                    display: flex;
                    justify-content: center; /* Teks diletakkan persis di tengah */
                    align-items: center;
                    box-sizing: border-box;
                    box-shadow: 0px 1px 2px rgba(0,0,0,0.05); 
                '>
                    <strong>{jenis}</strong>
                </div>
                """, 
                unsafe_allow_html=True
            )

    st.write("---")
    
    # Tombol Aksi
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Lihat Detail Mobil"):
            st.session_state.return_to = 'roc_ranking'
            move_to('detail_mobil')
    with col_b:
        if st.button("Ganti ke Pembobotan Manual"):
            move_to('manual_input')
    with col_c:
        if st.button("Hitung Rekomendasi 🚀", type="primary"):
            if len(urutan) == 7:
                st.session_state.bobot_kriteria = hitung_roc(urutan)
                st.session_state.metode_dipakai = "ROC (Rank Order Centroid)"
                move_to('hitung_hasil')
            else:
                st.error("Urutkan ke-7 kriteria terlebih dahulu sebelum menghitung!")
    
    st.divider()
    if st.button("⬅️ Kembali ke Landing Page"):
        move_to('landing')

#  3. Halaman pembobotan manual kriteria
elif st.session_state.page == 'manual_input':
    st.title("Halaman Pembobotan Manual")
    st.write("Masukkan nilai bobot (0.0 - 1.0) untuk setiap kriteria.")
    
    # Fitur Tombol Keterangan Benefit & Cost menggunakan Expander
    with st.expander("ℹ️ **Klik di sini untuk melihat Keterangan Benefit & Cost**"):
        st.markdown("""
        Berikut penjelasan tentang kriteria benefit dan cost:
        * **Kriteria Benefit (Keuntungan)**: Kriteria yang semakin **besar** nilainya, semakin **baik** untuk pengguna. 
          *Contoh: Engine Capacity, Horsepower, Total Speed, Seats, Torque.*
        * **Kriteria Cost (Biaya/Pengorbanan)**: Kriteria yang semakin **kecil** nilainya, semakin **baik** untuk pengguna.
          *Contoh: Cars Prices (Harga yang murah lebih baik), Performance (Waktu akselerasi yang lebih singkat lebih baik).*
        """)
    
    st.write("") # Spasi kosong

    # Form Input Bobot Manual
    c1, c2 = st.columns(2)
    with c1:
        w1 = st.number_input("Cars Prices (Cost)", value=0.0, format="%.4f")
        w2 = st.number_input("Total Speed (Benefit)", value=0.0, format="%.4f")
        w3 = st.number_input("Engine Capacity (Benefit)", value=0.0, format="%.4f")
    with c2:
        w4 = st.number_input("Horsepower (Benefit)", value=0.0, format="%.4f")
        w5 = st.number_input("Performance (Cost)", value=0.0, format="%.4f")
        w6 = st.number_input("Seats (Benefit)", value=0.0, format="%.4f")
        w7 = st.number_input("Torque (Benefit)", value=0.0, format="%.4f")

    tempat_notifikasi = st.empty()

    st.write("---")
    
    # Tombol Aksi
    col_x, col_mid, col_y = st.columns(3)
    
    with col_x:
        if st.button("Lihat Detail Mobil", key="btn_view_manual"):
            st.session_state.return_to = 'manual_input' 
            move_to('detail_mobil')
            
    with col_mid:
        if st.button("Ganti ke Ranking Prioritas (ROC)"):
            move_to('roc_ranking')
            
    with col_y:
        if st.button("Hitung Rekomendasi 🚀", type="primary"):
            
            # 1. Hitung total semua bobot yang diinput user
            total_bobot = w1 + w2 + w3 + w4 + w5 + w6 + w7
            semua_bobot = [w1, w2, w3, w4, w5, w6, w7]
            
            # 2. Cek apakah user membiarkan semuanya 0
            if total_bobot == 0.0:
                tempat_notifikasi.error("⚠️ Gagal memproses! Total bobot tidak boleh 0. Harap isi minimal satu nilai bobot.")
            
            # 3. Cek apakah ada input minus (negatif)
            elif any(bobot < 0.0 for bobot in semua_bobot):
                tempat_notifikasi.error("⚠️ Gagal memproses! Nilai bobot tidak boleh kurang dari 0.0.")
            
            # 4. Jika aman, lakukan NORMALISASI OTOMATIS dan simpan
            else:
                if total_bobot != 1.0:
                    w1 = w1 / total_bobot
                    w2 = w2 / total_bobot
                    w3 = w3 / total_bobot
                    w4 = w4 / total_bobot
                    w5 = w5 / total_bobot
                    w6 = w6 / total_bobot
                    w7 = w7 / total_bobot
                
                # Simpan bobot yang sudah dinormalisasi ke dalam sistem
                st.session_state.bobot_kriteria = {
                    "Cars Prices": w1, "Total Speed": w2, "Engine Capacity": w3,
                    "Horsepower": w4, "Performance": w5, "Seats": w6, "Torque": w7
                }
                st.session_state.metode_dipakai = "Manual (Subjektif)"
                move_to('hitung_hasil')

    st.divider()
    if st.button("⬅️ Kembali ke Menu Utama"):
        move_to('landing')


#  4. Halaman daftar detail mobil 
elif st.session_state.page == 'detail_mobil':
    st.title("Detail Spesifikasi Mobil")
    
    if st.session_state.df_mentah is not None:
        st.write(f"Total data: {len(st.session_state.df_mentah)} mobil")
        st.dataframe(st.session_state.df_mentah, use_container_width=True)
    else:
        st.warning("Data kosong. Silakan kembali ke halaman utama untuk meng-upload data.")
    
    st.divider()
    
    # Ambil tujuan kembali, default ke 'landing' jika belum ada data
    target_kembali = st.session_state.get('return_to', 'landing')
    
    if st.button("⬅️ Kembali ke Halaman Sebelumnya"):
        move_to(target_kembali)

#  5. Halaman hitung dan hasil perangkingan alternatif mobil terbaik  
elif st.session_state.page == 'hitung_hasil':
    st.title("Hasil Rekomendasi Pemilihan Mobil")
    
    if st.session_state.df_mentah is not None and st.session_state.bobot_kriteria:
        with st.spinner("Sedang memproses data dan menghitung..."):
            # PASTIKAN TULISANNYA BEGINI:
            cols_to_use = ["Engine Capacity", "Horsepower", "Total Speed", "Performance", "Cars Prices", "Seats", "Torque"]
            
            # Baris yang error kemarin akan aman sekarang
            df_input = st.session_state.df_mentah[cols_to_use] 
            
            # 1. Preprocessing
            df_bersih = preprocess_data(df_input)
            
            # 2. Perhitungan EDAS
            skor_as = hitung_edas(df_bersih, st.session_state.bobot_kriteria)
            
            # 3. Gabungkan Hasil
            df_hasil = st.session_state.df_mentah.loc[df_bersih.index].copy()
            df_hasil['Appraisal Score (AS)'] = skor_as
            df_hasil = df_hasil.sort_values(by='Appraisal Score (AS)', ascending=False)
            
            st.success("✅ Perhitungan Selesai!")
            
            #  Menampilkan Informasi Metode yang Dipakai 
            st.info(f"Metode Pembobotan yang digunakan: **{st.session_state.get('metode_dipakai', 'Tidak Diketahui')}**")
            
            # Tampilkan nilai bobotnya agar user bisa memastikan (bisa dihapus nanti kalau menuhi layar)
            with st.expander("Lihat Nilai Bobot Kriteria yang Digunakan"):
                st.write(st.session_state.bobot_kriteria)
            
            st.write("---")
            st.subheader("🏆 Tabel Ranking Alternatif Mobil Terbaik")
            
            # Reset index agar rankingnya mulai dari 1 di tabel
            df_hasil_tampil = df_hasil.reset_index(drop=True)
            df_hasil_tampil.index = df_hasil_tampil.index + 1 
            
            st.dataframe(df_hasil_tampil, use_container_width=True)
            
    else:
        st.error("Data atau Bobot belum siap. Silakan kembali ke halaman sebelumnya.")

    st.divider()
    if st.button("🔄 Ulangi dari Awal"):
        # 1. Kembalikan penanda ke default
        st.session_state.is_custom_data = False
        
        # 2. Muat ulang dataset bawaan sistem
        st.session_state.df_mentah = load_default_data()
        
        # 3. Kosongkan bobot kriteria sebelumnya agar tidak mempengaruhi perhitungan selanjutnya
        st.session_state.bobot_kriteria = {}
        
        # 4. Pindah ke halaman utama
        move_to('landing')