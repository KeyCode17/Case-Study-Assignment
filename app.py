# Streamlit Core
import streamlit as st
import streamlit_nested_layout
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components

import os
import dill
import py7zr
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

dirloc = os.getcwd()

# Define file paths
csv_path = os.path.join(dirloc, 'bahan_cleaned.csv')
archive_path = os.path.join(dirloc, 'bahan_cleaned.7z')

# Check if the CSV file exists. If not, try to extract it from the 7z archive.
if not os.path.exists(csv_path):
    print(f"'{csv_path}' not found. Checking for archive...")
    
    # Check if the 7z archive exists
    if os.path.exists(archive_path):
        print(f"Archive '{archive_path}' found. Extracting...")
        
        # Extract the 7z archive
        with py7zr.SevenZipFile(archive_path, 'r') as archive:
            archive.extractall(path=dirloc)

        df_cleaned = pd.read_csv(csv_path)
        print(f"Successfully extracted archive to '{dirloc}'.")
    else:
        print(f"No CSV file or archive found. Please ensure either '{csv_path}' or '{archive_path}' exists.")
else:
    df_cleaned = pd.read_csv(csv_path)
    print(f"CSV file '{csv_path}' already exists. No need to extract from archive.")

st.title("Case Study Assignment (Airline)")


main_menu = option_menu(None, ["Home", "Data Cleaning", "EDA"], 
    icons=['bi-house-fill',  'bi-droplet-half', 'bi-clipboard2-data'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "nav-link": {"font-size": "1.5rem", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"font-size": "1.45rem"}
    })

st.query_params["Page"]=main_menu

if st.query_params["Page"] == "Home":
    st.subheader("Mochammad Daffa Putra karyudi", divider="gray")
    st.markdown('''
    - **Nickname**: Karyudi
    - **Email**: [m.daffa.karyudi@gmail.com](mailto:m.daffa.karyudi@gmail.com)
    - **LinkedIn**: [Daffa Karyudi](https://www.linkedin.com/in/daffakaryudi)
    ''')
    with st.expander("Limitation Analysis", expanded=False):
        st.markdown("""
        ## Limitation
Meskipun analisa ini didasarkan pada dataset yang kaya, terdapat beberapa keterbatasan fundamental yang harus dipertimbangkan saat mengevaluasi temuan. Keterbatasan utama berasal dari ketiadaan **kamus data (data dictionary)** atau dokumentasi teknis yang menyertainya. Berdasarkan struktur dan nama kolom (misalnya, `pax_name`, `travel_date`, `airline`, `harga`, `class`, `sector`), data ini sangat diindikasikan berasal dari domain industri penerbangan. Namun, tanpa definisi operasional yang formal, analisis bergantung pada asumsi yang berpotensi signifikan, yang termanifestasi dalam keterbatasan berikut:

1.  **Ambiguitas Kritis pada Variabel Kunci (`class`):**
    * **Variabel `class`:** Kolom ini berisi kode-kode seperti **'X', 'Q', 'V', 'L', 'T', 'P'**, dll. Dalam konteks penerbangan, ini kemungkinan besar adalah *Fare Classes* atau *Reservation Booking Designators (RBDs)*. Tanpa metadata, mustahil untuk mengetahui secara pasti:
        * **Hirarki Kabin:** Kelas mana yang termasuk dalam kategori *First Class*, *Business*, *Premium Economy*, atau *Economy*.
        * **Aturan Tarif (Fare Rules):** Apakah sebuah kelas tiket dapat di-refund, diubah jadwalnya, atau berapa banyak bagasi yang diizinkan.
        * **Jenis Tarif:** Apakah ini tiket berbayar (revenue ticket) atau tiket penukaran poin (award/redemption ticket).

2.  **Ketidakpastian dalam Hubungan Ordinal dan Kuantitatif:**
    * Hubungan antara **`class`** dan **`harga`** menjadi tidak pasti. Secara teoretis dalam industri penerbangan, kelas tarif yang berbeda memiliki tingkat harga yang berbeda pula. Namun, tanpa definisi, kita tidak dapat memvalidasi apakah kelas 'Y' secara konsisten lebih mahal daripada kelas 'Q'. Analisis apa pun yang mencoba membangun hubungan antara kelas dan harga harus didasarkan pada asumsi yang diturunkan dari data itu sendiri (misalnya, dengan menghitung harga rata-rata per kelas), bukan dari aturan bisnis yang telah ditetapkan, sehingga mengurangi validitas kesimpulan.

3.  **Keterbatasan pada Kedalaman Analisis Strategis:**
    * Ketiadaan metadata menghalangi analisis strategis yang mendalam, seperti:
        * **Analisis *Yield Management*:** Tidak mungkin secara akurat menganalisis strategi harga maskapai atau *yield* per sektor jika kita tidak dapat membedakan antara tiket diskon dan tiket tarif penuh.
        * **Segmentasi Pelanggan:** Upaya untuk melakukan segmentasi penumpang berdasarkan kombinasi `class`, `agent_type`, dan `sector` akan bersifat dangkal karena makna sebenarnya dari segmen-segmen ini tidak diketahui.
        * **Analisis *Booking Window*:** Menghitung rentang waktu antara `generation_date` dan `travel_date` adalah mungkin, tetapi menghubungkannya dengan perilaku pemesanan menjadi sulit tanpa mengetahui jenis tarif (`class`) yang dibeli.

4.  **Risiko dalam Pra-pemrosesan dan Pemodelan:**
    * Keputusan untuk melakukan *encoding* pada variabel `class` menjadi sangat problematis. Memperlakukannya sebagai variabel nominal murni (melalui *one-hot encoding*) akan mengabaikan potensi hirarki harga, sementara memaksakan urutan ordinal akan sepenuhnya didasarkan pada spekulasi dan berisiko memasukkan bias yang signifikan ke dalam model prediktif.

Dengan demikian, temuan dari analisa ini harus dianggap sebagai eksplorasi awal terhadap pola-pola yang ada dalam data. Untuk analisis yang lebih definitif dan kesimpulan yang dapat ditindaklanjuti secara komersial, diperlukan validasi dan pengayaan data dengan kamus data resmi dari penyedia data.
        """)

elif st.query_params["Page"] == "Data Cleaning":
    st.subheader("Data Cleaning", divider="gray")
    
    # Display basic statistics of the cleaned data
    with st.expander('Tahapan Data Cleaning', expanded=False):
        st.markdown("""
        #### Dokumentasi Teknis: Pipeline Pembersihan Data Airline `clean_airline_dataset_comprehensive`

---

##### Fase 1: Standarisasi Awal Nilai Region
* **Tujuan:** Mengidentifikasi dan menetralkan nilai pada kolom `region` yang secara definitif tidak valid atau merupakan placeholder.
* **Asumsi yang Mendasari:** Nilai-nilai seperti `-1`, `Indonesia`, `nan`, dan `NULL` tidak mengandung informasi geografis yang spesifik dan dapat dianalisis. Nilai-nilai ini dianggap sebagai hasil dari kesalahan input atau placeholder sistem, yang secara fungsional setara dengan data yang hilang (*missing value*).
* **Langkah Eksekusi:**
    1.  Skrip mengidentifikasi `region` dengan nilai-nilai dari daftar `invalid_region_identifiers`.
    2.  Jumlah kemunculan setiap nilai tidak valid didokumentasikan melalui logging.
    3.  Semua nilai yang teridentifikasi diubah menjadi `np.nan` (NULL) untuk diproses lebih lanjut pada fase imputasi.

##### Fase 2: Pemformatan String Region
* **Tujuan:** Menyeragamkan format penulisan untuk semua data `region` yang tersisa guna memastikan konsistensi kategorikal.
* **Asumsi yang Mendasari:** Konsistensi format penulisan (tanpa spasi berlebih dan menggunakan "Title Case") adalah prasyarat untuk pemetaan dan pengelompokan yang akurat. Tanpa ini, nilai seperti `'jawa barat'` dan `'Jawa Barat'` akan dianggap sebagai dua kategori yang berbeda.
* **Langkah Eksekusi:**
    1.  Menghapus spasi di awal dan akhir string (`.str.strip()`).
    2.  Mengganti spasi ganda atau lebih di tengah string dengan satu spasi (`.str.replace(r'\s+', ' ', regex=True)`).
    3.  Mengonversi seluruh string ke format "Title Case" (`.str.title()`).

##### Fase 3: Implementasi Pemetaan Region Indonesia
* **Tujuan:** Menerapkan aturan pemetaan yang kompleks untuk mengonsolidasikan berbagai variasi nama region ke standar nama provinsi.
* **Asumsi yang Mendasari:** File `indonesian_region_mapping_v1.1.pkl` adalah "sumber kebenaran" (*single source of truth*). Kamus di dalamnya, yang dibuat dari analisis sebelumnya, secara akurat memetakan variasi nama (termasuk singkatan, kesalahan ketik, dan nama level kota) ke nama provinsi yang standar.
* **Langkah Eksekusi:**
    1.  Pipeline memuat kamus pemetaan dari file `.pkl` yang telah disiapkan.
    2.  Fungsi `standardize_region` diaplikasikan ke setiap baris pada kolom `region`.
    3.  Fungsi ini mencocokkan nilai region dengan kunci dalam kamus. Jika ditemukan, nilainya akan diperbarui. Jika tidak, nilai asli dipertahankan.

##### Fase 4: Strategi Imputasi Region yang Hilang
* **Tujuan:** Mengisi nilai `region` yang kosong (`np.nan`) secara cerdas menggunakan informasi dari data itu sendiri.
* **Asumsi yang Mendasari:** Perilaku bisnis seorang agen tiket (`agent`) bersifat konsisten secara geografis. Oleh karena itu, modus (region yang paling sering muncul) dari data historis seorang agen adalah prediktor yang paling andal untuk mengisi data region yang hilang pada transaksi lain dari agen yang sama.
* **Langkah Eksekusi:**
    1.  Skrip membuat pemetaan dinamis dengan mengelompokkan data berdasarkan `agent` dan mencari modus `region` untuk setiap agen.
    2.  Fungsi `apply_region_imputation` diterapkan. Jika sebuah baris memiliki `region` kosong, fungsi ini mengisinya dengan modus region dari `agent` yang bersangkutan.
    3.  Jika modus tidak ditemukan, region diisi dengan nilai `'Unknown Region'` untuk menandakan bahwa imputasi tidak memungkinkan.

##### Fase 5: Pemrosesan Komprehensif `class` Penerbangan
* **Tujuan:** Membersihkan kolom `class` yang sangat terpolusi melalui kombinasi penghapusan data yang tidak dapat diperbaiki, pemetaan variasi, dan imputasi cerdas.
* **Asumsi yang Mendasari:**
    * **Untuk Penghapusan:** Entri seperti `Promo`, `DEPOSIT`, `-`, dan kode acak lainnya bukan merupakan *fare class* yang valid. Nilai-nilai ini adalah *noise* yang tidak dapat diselamatkan, dan menghapus seluruh baris data adalah pendekatan yang lebih baik daripada melakukan imputasi yang sangat spekulatif.
    * **Untuk Pemetaan:** Variasi seperti `v`, `A/0`, dan `E1` adalah representasi non-standar namun valid dari kode IATA induknya dan dapat dikonsolidasikan tanpa kehilangan makna.
    * **Untuk Imputasi:** Terdapat korelasi kuat antara `harga` tiket dan `class`-nya. Oleh karena itu, harga adalah fitur terbaik yang tersedia untuk mengestimasi `class` yang hilang, terutama jika dikombinasikan dengan `airline`. Kelas `'Y'` adalah *fallback* yang paling umum dan aman.
* **Langkah Eksekusi:**
    1.  **Fase 5a (Penghapusan Ketat):** Memuat daftar `entries_to_delete` dan menghapus seluruh baris data yang `class`-nya cocok dengan salah satu entri dalam daftar tersebut.
    2.  **Fase 5b (Pemetaan Standarisasi):** Menerapkan kamus `flight_class_mapping` untuk menstandarisasi variasi `class` yang valid ke kode IATA induknya.
    3.  **Fase 5c (Imputasi Berbasis Harga):** Untuk `class` yang masih kosong, pipeline mengisinya dengan mencari kelas pada maskapai yang sama yang memiliki median harga paling mendekati.

##### Fase 6: Eliminasi Data Duplikat
* **Tujuan:** Menghapus baris data yang identik secara absolut untuk memastikan setiap baris unik.
* **Asumsi yang Mendasari:** Baris data yang 100% identik adalah hasil dari kesalahan teknis dalam proses pengumpulan data (misalnya, duplikasi saat proses ETL) dan tidak mengandung informasi baru yang bernilai.
* **Langkah Eksekusi:**
    1.  Fungsi `df.drop_duplicates()` dipanggil pada keseluruhan dataset.
    2.  Jumlah baris duplikat yang dihapus dicatat untuk dokumentasi.

##### Fase 7: Validasi Integritas Data
* **Tujuan:** Melakukan pemeriksaan kewajaran (*sanity check*) akhir untuk memastikan dataset mematuhi aturan bisnis yang fundamental.
* **Asumsi yang Mendasari:** Transaksi penerbangan yang valid secara logis harus memiliki harga positif dan tanggal perjalanan yang tidak mendahului tanggal pemesanan. Nilai di luar batas kewajaran (seperti `booking_lead_time > 365 hari`) dianggap sebagai anomali.
* **Langkah Eksekusi:**
    1.  Kolom sementara `booking_lead_time` dibuat.
    2.  Dataset diperiksa untuk kondisi-kondisi yang melanggar aturan bisnis (misal: `harga < 0`).
    3.  Jumlah pelanggaran dicatat dalam log sebagai indikator kesehatan data akhir.
    4.  Kolom sementara dihapus setelah validasi selesai.
        """)

    # Display the first few rows of the cleaned data
    st.write("Cleaned Data Sample:")
    st.dataframe(df_cleaned.head())


elif st.query_params["Page"] == "EDA":
    st.header("Exploratory Data Analysis (EDA)", divider="gray")
    option = st.selectbox(
        "Choose Section",
        ("Q1: Bagaimana peringkat 10 maskapai dengan distribusi pesanan terbanyak?", "Q2: Bagaimana peringkat 10 besar maskapai berdasarkan rata-rata harga tiket?",
        "Q3: Wilayah mana yang menghasilkan pemesanan terbanyak?", "Q4: Bagaimana pola pemesanan berdasarkan tanggal pemberangkatan?",
        "Q5: Bagaimana tren pemesanan berdasarkan tanggal pemesanannya?", "Q6: Kelas penerbangan mana yang paling populer?",
        "Q7: Apa saja 10 rute penerbangan terpopuler?", "Q8: Apa saja 10 maskapai dengan pendapatan tertinggi?",
        "Q9: Bagaimana perbedaan pola pemesanan berdasarkan hari dalam seminggu?","Ringkasan Analisis & Implikasi Strategis (Q1-Q9)")
    )

    st.query_params["Section"]=option



    if st.query_params["Section"] == "Q1: Bagaimana peringkat 10 maskapai dengan distribusi pesanan terbanyak?":
        st.subheader(option)
        
        # Calculate distribution of orders by airline (Top 10)
        airline_dist = df_cleaned['airline'].value_counts().head(10).reset_index()
        airline_dist.columns = ['airline', 'order_count']
        
        # Create pie chart
        fig1 = px.pie(airline_dist, values='order_count', names='airline')
        with st.expander('Hasil Analisis', expanded=False):
            st.markdown("""
            #### Wawasan dan Interpretasi (Insights)

1.  **Dominasi Pasar yang Jelas oleh Satu Maskapai (Market Dominance)**
    * Maskapai **XG** mendominasi pasar secara signifikan dengan pangsa **36.3%**. Ini menunjukkan bahwa lebih dari sepertiga dari total pesanan di antara 10 maskapai teratas dikuasai oleh satu pemain. Hal ini bisa disebabkan oleh berbagai faktor seperti jaringan rute yang luas, harga yang kompetitif, kapasitas penerbangan yang besar, atau loyalitas pelanggan yang tinggi.

2.  **Persaingan Ketat di Peringkat Kedua (Tier 2 Competition)**
    * Maskapai **AZ** dan **LE** bersaing sangat ketat di posisi kedua, masing-masing memegang pangsa pasar **14.6%**. Jika digabungkan, kekuatan mereka (29.2%) hampir menyaingi pemimpin pasar. Ini menandakan adanya persaingan yang sehat dan tidak ada monopoli tunggal di level ini. Strategi mereka kemungkinan besar sangat mirip, baik dari segi harga maupun target pasar.
        
3.  **Pemain Menengah yang Solid (Mid-Tier Players)**
    * Maskapai **MI (10.6%)** dan **GF (8.23%)** merupakan pemain menengah yang solid. Mereka memiliki pangsa pasar yang cukup untuk tetap relevan dan kemungkinan besar memiliki ceruk pasar (niche market) atau keunggulan di rute-rute tertentu.

4.  **Pemain Ceruk dan Pelengkap (Niche & Minor Players)**
    * Maskapai **HV, TI, DB, QT, dan TS** secara kolektif hanya menyumbang sekitar **15.7%** dari total pesanan. Mereka adalah pemain yang lebih kecil dalam konteks 10 besar ini. Kemungkinan mereka fokus pada:
        * **Rute spesifik** yang tidak banyak dilayani oleh maskapai besar.
        * **Model bisnis tertentu** (misalnya, LCC - Low-Cost Carrier atau maskapai regional).
        * **Target demografis** yang lebih sempit.

---

#### Kesimpulan Analisis

Distribusi pesanan ini menunjukkan struktur pasar yang **terkonsentrasi di puncak (Top-Heavy Market)**. Satu maskapai (XG) adalah pemimpin yang tak terbantahkan, diikuti oleh dua pesaing kuat (AZ & LE), dan sisanya adalah pemain menengah atau ceruk.
            """)

        # Display the chart in Streamlit
        st.plotly_chart(fig1)
        
        # Optionally display the data in a table with percentages
        with st.expander('Data Tabel', expanded=False):
            total_orders = airline_dist['order_count'].sum()
            airline_dist['percentage'] = (airline_dist['order_count'] / total_orders * 100).round(2).astype(str) + '%'
            st.write("Data table with percentages:")
            st.dataframe(airline_dist, hide_index=True)
        



    elif st.query_params["Section"] == "Q2: Bagaimana peringkat 10 besar maskapai berdasarkan rata-rata harga tiket?":
        # Question 2: Average ticket prices by airline (Top 10)
        st.subheader(option)

        avg_price_airline = df_cleaned.groupby('airline')['harga'].mean().reset_index()

        # Sort by average price from highest to lowest and get top 10
        avg_price_airline = avg_price_airline.sort_values('harga', ascending=False).head(10)

        # Create horizontal bar chart
        fig2 = px.bar(avg_price_airline, x='harga', y='airline', orientation='h')

        # Update layout to improve readability
        fig2.update_layout(
            yaxis={'categoryorder': 'total ascending'},  # This ensures the sorting is maintained
            xaxis_title="Average Price (Harga)",
            yaxis_title="Airline",
            width=1000,  # Increase width
            height=600   # Reduced height since we only have 10 bars now
        )

        with st.expander('Hasil Analisis', expanded=False):
            st.markdown("""
            #### Observasi dan Interpretasi

**1. Observasi Teknis Penting: Harga Ternormalisasi**
Sebelum masuk ke interpretasi bisnis, penting untuk dicatat bahwa nilai pada kolom `harga` tampaknya telah **dinormalisasi atau diskalakan** (menghasilkan nilai desimal yang sangat kecil). Ini adalah praktik umum dalam pengolahan data. Artinya, kita sedang membandingkan **nilai relatif** antar maskapai, bukan harga absolut dalam Rupiah. Maskapai `KD` memiliki harga rata-rata hampir dua kali lipat lebih tinggi dari `LC` dalam skala ini.

**2. Wawasan Paling Krusial: Perbandingan dengan Volume Pesanan (Q1)**
Inilah temuan yang paling signifikan: **Tidak ada satu pun maskapai dari daftar 10 teratas berdasarkan volume pesanan (Q1) yang muncul di daftar 10 teratas berdasarkan harga rata-rata tertinggi ini.**

* **Maskapai Populer (Q1):** XG, AZ, LE, MI, dll. (volume tinggi).
* **Maskapai Mahal (Q2):** KD, BO, WZ, PC, dll. (harga tinggi).

Ini dengan jelas menunjukkan adanya **dua strategi bisnis yang berbeda** di pasar:
* **Strategi Volume Tinggi, Harga Rendah:** Maskapai seperti **XG, AZ, dan LE** kemungkinan besar adalah *Low-Cost Carrier (LCC)* atau pemain besar yang fokus pada kelas ekonomi untuk menarik jumlah penumpang sebanyak mungkin.
* **Strategi Volume Rendah, Harga Tinggi:** Maskapai dalam daftar ini (**KD, BO, WZ**, dst.) kemungkinan adalah *Full-Service Carrier*, maskapai premium, atau maskapai yang melayani rute-rute khusus/bisnis di mana mereka bisa menetapkan harga yang lebih tinggi.

---

#### Hipotesis

Berdasarkan temuan ini, merumuskan beberapa hipotesis yang perlu divalidasi:

1.  **Hipotesis:** Maskapai seperti `KD` dan `BO` memiliki proporsi penjualan tiket kelas **bisnis/utama** yang lebih tinggi.
2.  **Hipotesis:** Maskapai seperti `XG` dan `AZ` (dari Q1) hampir secara eksklusif menjual tiket kelas **ekonomi**.

---

#### Kesimpulan
Analisis ini mengungkap adanya segmen pasar premium yang dilayani oleh sekelompok maskapai tertentu. Pemosisian harga mereka secara signifikan lebih tinggi daripada rata-rata pasar, menandakan strategi bisnis yang berfokus pada nilai dan kualitas, bukan pada volume.
            """)
        # Display the chart in Streamlit
        st.plotly_chart(fig2)

        with st.expander('Data Tabel', expanded=False):
            st.write("Data table with average prices:")
            st.dataframe(avg_price_airline, hide_index=True)



    elif st.query_params["Section"] == "Q3: Wilayah mana yang menghasilkan pemesanan terbanyak?":
        # Question 3: Orders by region (Top 10)
        st.subheader(option)

        with st.expander('Hasil Analisis', expanded=False):
            st.markdown("""
            #### Temuan Utama

1. Temuan Utama (Key Findings):
Dominasi Absolut Satu Wilayah: DKI Jakarta adalah pusat pasar yang dominan secara absolut, dengan total 1.72 juta pemesanan. Angka ini menciptakan jarak yang sangat besar dengan wilayah peringkat kedua, menunjukkan peran Jakarta sebagai hub utama yang tak tertandingi.

2. Pasar Sekunder yang Signifikan: Meskipun ada jarak yang jauh, Jawa Barat (282 ribu), Sumatera Utara (192 ribu), Kalimantan Timur (144 ribu), dan Jawa Timur (107 ribu) merupakan pasar sekunder yang sangat penting dan merepresentasikan kantong-kantong permintaan yang kuat.

3. Pentingnya Validitas Data: Terdapat kategori "Unknown Region" dengan volume pemesanan yang cukup tinggi (96 ribu). Hal ini menunjukkan adanya potensi kehilangan informasi yang dapat memengaruhi akurasi pemetaan geografis secara keseluruhan. Perbaikan kualitas pencatatan data diperlukan.

4. Distribusi Ekor Panjang (Long Tail): Setelah 5-10 wilayah teratas, volume pemesanan per wilayah menurun secara drastis, menunjukkan adanya banyak wilayah dengan kontribusi yang lebih kecil (ekor panjang).
___

#### Kesimpulan
Distribusi pasar secara geografis sangat terkonsentrasi di DKI Jakarta. Namun, terdapat beberapa pasar sekunder yang kuat yang menjadi pilar penting bagi bisnis. Peningkatan kualitas data regional akan semakin mempertajam pemahaman tentang lanskap pasar ini.
            """)

        region_bookings = df_cleaned['region'].value_counts().reset_index()
        region_bookings.columns = ['region', 'bookings']

        # Sort by bookings from highest to lowest
        region_bookings = region_bookings.sort_values('bookings', ascending=False)

        fig3 = px.bar(region_bookings, x='region', y='bookings')

        # Correct way to update x-axis properties
        fig3.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig3)

        with st.expander('Data Tabel', expanded=False):
            st.write("Data table with bookings by region:")
            st.dataframe(region_bookings, hide_index=True)



    elif st.query_params["Section"] == "Q4: Bagaimana pola pemesanan berdasarkan tanggal pemberangkatan?":
        # Question 4: Booking patterns by travel date
        st.subheader(option)

        with st.expander('Hasil Analisis', expanded=False):
            st.markdown("""
            #### **Temuan Utama**

Analisis data time-series menunjukkan empat fase yang berbeda dalam siklus hidup pemesanan selama periode yang diamati:

1. **Fase Permulaan (Januari 2017 - Maret 2017):**
   * Aktivitas pemesanan sangat rendah dan sporadis. Periode ini kemungkinan menandai awal dari pengumpulan data atau fase awal peluncuran bisnis di mana volume masih sangat kecil.

2. **Fase Pertumbuhan dan Puncak (April 2017 - Desember 2018):**
   * Dimulai pada April 2017, terjadi lonjakan volume pemesanan yang signifikan, menandai dimulainya aktivitas bisnis yang sesungguhnya.
   * Terdapat **tren pertumbuhan yang jelas** dari pertengahan 2017 hingga akhir 2018.
   * **Pola musiman (seasonality)** sangat terlihat jelas, dengan puncak-puncak permintaan yang dapat diidentifikasi:
     * **Puncak Pertengahan Tahun (Juni 2018):** Lonjakan permintaan yang sangat kuat terjadi di sekitar bulan Juni 2018. Ini sangat mungkin berkorelasi dengan periode libur **Lebaran (Idul Fitri)**, yang merupakan musim puncak perjalanan domestik di Indonesia.
     * **Puncak Akhir Tahun (Desember 2018):** Titik tertinggi absolut dari seluruh periode data terjadi pada akhir Desember 2018. Puncak ini secara jelas berhubungan dengan musim liburan **Natal dan Tahun Baru**, dengan tanggal perjalanan tertinggi pada **21 Desember 2018 (13.862 pemesanan)**.

3. **Fase Penurunan Drastis (Januari 2019 - Maret 2019):**
   * Segera setelah mencapai puncaknya di awal Januari 2019, volume pemesanan mengalami **penurunan yang sangat tajam dan drastis**. Ini bukan sekadar penurunan musiman biasa, melainkan sebuah "crash" yang menandakan adanya perubahan struktural.
   * Volume jatuh dari ribuan pemesanan per hari menjadi hanya ratusan dalam waktu singkat.

4. **Fase Aktivitas Rendah (April 2019 dan seterusnya):**
   * Setelah penurunan drastis, volume pemesanan stabil pada tingkat yang sangat rendah, jauh di bawah level sebelum puncak. Ini menunjukkan "kondisi normal baru" bagi bisnis atau data yang tercatat.

---
#### **Kesimpulan dan Implikasi Strategis**

* Bisnis ini menunjukkan pola musiman yang dapat diprediksi selama periode puncaknya (2017-2018), terutama terkait dengan libur Lebaran dan Akhir Tahun. Ini adalah informasi krusial untuk perencanaan kapasitas, stok, dan strategi penetapan harga.
* **Peristiwa paling kritikal** dalam data ini adalah **penurunan drastis pada awal tahun 2019**. Ini adalah anomali yang paling signifikan dan memerlukan investigasi mendalam. Kemungkinan penyebabnya bisa meliputi:
  * **Faktor Eksternal:** Krisis ekonomi, bencana alam, atau peristiwa besar lain yang menekan permintaan perjalanan.
  * **Perubahan Internal:** Perubahan strategi bisnis, penghentian rute/layanan populer, atau perubahan model harga.
  * **Isu Teknis:** Masalah pada sistem pengumpulan data yang menyebabkan data tidak tercatat dengan benar setelah Januari 2019.

Tanpa memahami penyebab penurunan ini, setiap upaya peramalan (forecasting) di masa depan akan sangat tidak akurat.
            """)
        
        travel_date_pattern = df_cleaned.groupby('travel_date').size().reset_index(name='bookings')
        fig4 = px.line(travel_date_pattern, x='travel_date', y='bookings')
        st.plotly_chart(fig4)
        with st.expander('Data Tabel', expanded=False):
            st.write("Data table with bookings by travel date:")
            st.dataframe(travel_date_pattern, hide_index=True)



    elif st.query_params["Section"] == "Q5: Bagaimana tren pemesanan berdasarkan tanggal pemesanannya?":
        # Question 5: Booking trends by booking date
        st.subheader(option)

        with st.expander('Hasil Analisis', expanded=False):
            st.markdown("""
            #### **Temuan Utama**

1.  **Pola Mingguan yang Sangat Kuat (Strong Weekly Cyclicality):**
    * Grafik menunjukkan pola naik-turun yang sangat teratur dan berulang setiap minggu. Volume pemesanan secara konsisten **tinggi pada hari kerja (Senin-Jumat)** dan **turun secara signifikan pada akhir pekan (Sabtu-Minggu)**.
    * Pola ini sangat umum untuk bisnis yang melayani segmen korporat atau B2B, di mana keputusan pembelian sebagian besar terjadi selama jam kerja. Ini juga bisa mengindikasikan bahwa kampanye pemasaran atau aktivitas tim penjualan paling aktif pada hari kerja.

2.  **Periode Penjualan Puncak (Peak Sales Periods):**
    * Terdapat beberapa periode di mana volume pemesanan harian melonjak secara masif, jauh di atas rata-rata normal. Puncak tertinggi terjadi pada **akhir 2018**, khususnya di **Desember**.
    * **Puncak Tertinggi (5 Desember 2018):** Pada tanggal ini, tercatat **21.941 pemesanan**, yang merupakan aktivitas penjualan harian tertinggi dalam seluruh dataset. Puncak-puncak lain yang signifikan juga terjadi di sekitar periode ini, seperti **7 Desember 2018 (21.298 pemesanan)**.
    * Lonjakan luar biasa ini kemungkinan besar bukan disebabkan oleh permintaan organik, melainkan oleh **acara penjualan besar (major sales events)**, seperti promo akhir tahun, travel fair online, atau kampanye promosi besar-besaran.

3.  **Perbedaan Kunci dengan Tren Tanggal Perjalanan (Q4):**
    * **Booking Window:** Perbedaan antara grafik `generation_date` (kapan pesan) dan `travel_date` (kapan terbang) menunjukkan adanya "booking window" atau jeda waktu antara pemesanan dan perjalanan. Puncak pemesanan di awal Desember 2018 adalah untuk perjalanan di akhir Desember 2018 dan awal Januari 2019.
    * **Sifat Puncak:** Puncak pada `generation_date` lebih "tajam" dan sering kali didorong oleh promo (misalnya, promo gaji bulanan di tanggal 25-28 atau promo 10.10, 11.11, 12.12), sedangkan puncak pada `travel_date` lebih "lebar" dan mengikuti kalender liburan publik.
---
#### **Kesimpulan dan Implikasi Strategis**

* **Ritme Bisnis Didominasi Hari Kerja:** Strategi penjualan, alokasi sumber daya tim, dan jadwal kampanye email/digital marketing harus dioptimalkan untuk memaksimalkan momentum selama hari Senin hingga Jumat.
* **Keberhasilan Kampanye Penjualan Terukur:** Puncak-puncak masif pada `generation_date` membuktikan bahwa kampanye penjualan yang terfokus dan berbatas waktu sangat efektif dalam mendongkrak volume. Analisis lebih lanjut pada tanggal-tanggal puncak ini dapat memberikan wawasan tentang jenis promosi apa yang paling berhasil.
* **Manajemen Operasional:** Memahami pola mingguan ini krusial untuk manajemen staf. Tim layanan pelanggan dan operasional harus disiapkan untuk menangani volume yang lebih tinggi selama hari kerja.

Analisis `generation_date` ini memberikan pandangan langsung ke "inti" aktivitas penjualan harian perusahaan, melengkapi analisis musiman jangka panjang yang kita lihat dari `travel_date`.
            """)
        generation_trend = df_cleaned.groupby('generation_date').size().reset_index(name='bookings')
        fig5 = px.line(generation_trend, x='generation_date', y='bookings')
        st.plotly_chart(fig5)
        with st.expander('Data Tabel', expanded=False):
            st.write("Data table with bookings by generation date:")
            st.dataframe(generation_trend, hide_index=True)



    elif st.query_params["Section"] == "Q6: Kelas penerbangan mana yang paling populer?":
        # Question 6: Most popular flight classes
        st.subheader(option)

        with st.expander('Hasil Analisis', expanded=False):
            st.markdown("""
            #### **Temuan Utama**

1.  **Memahami Kode Kelas (Fare Class):** Penting untuk dipahami bahwa kode satu huruf ini (V, X, Q, dll.) bukanlah kelas kabin utama (Ekonomi, Bisnis, First), melainkan **Kelas Tarif** atau **Booking Code**. Setiap kelas tarif memiliki harga dan aturan yang berbeda (misal: tingkat fleksibilitas, jatah bagasi, perolehan miles), meskipun berada dalam kabin yang sama.

2.  **Dominasi Kelas Tarif Ekonomi Promo:**
    * Kelas tarif **V (370k)**, **X (323k)**, **Q (314k)**, **T (267k)**, dan **N (255k)** adalah yang paling populer dengan selisih yang signifikan.
    * Dalam industri penerbangan, kelas-kelas tarif ini secara universal merepresentasikan berbagai tingkatan **tiket Kelas Ekonomi dengan harga diskon atau promo**. Volume penjualan yang sangat tinggi pada kelas-kelas ini mengonfirmasi bahwa mayoritas besar pelanggan sangat sensitif terhadap harga.

3.  **Distribusi Ekor Panjang (Long-Tail Distribution):**
    * Setelah 10-15 kelas teratas, jumlah pemesanan untuk setiap kelas tarif menurun secara drastis.
    * Kelas-kelas dengan volume lebih rendah seperti **J, C, F, dan D** secara tradisional sering dikaitkan dengan **Kelas Bisnis atau First Class** yang harganya jauh lebih mahal dan fleksibel. Jumlahnya yang sedikit sangat sesuai dengan ekspektasi pasar.
---
#### **Kesimpulan dan Implikasi Strategis**

* **Model Bisnis Berbasis Volume Ekonomi:** Data ini adalah bukti terkuat bahwa mesin utama pendapatan (dari segi volume) berasal dari penjualan tiket Kelas Ekonomi, khususnya pada tingkatan harga yang paling rendah. Strategi penetapan harga dan manajemen inventaris untuk kelas-kelas ini adalah yang paling krusial.
* **Peluang Segmentasi:** Keragaman kelas tarif menunjukkan adanya upaya maskapai untuk melakukan segmentasi pasar. Meskipun volume kelas premium (seperti J atau C) kecil, yield atau keuntungan per tiketnya bisa jadi jauh lebih tinggi. Analisis lebih lanjut bisa menggabungkan data ini dengan harga untuk menghitung profitabilitas per kelas tarif.
* **Wawasan untuk Pemasaran:** Kampanye pemasaran yang menargetkan audiens luas harus fokus pada promosi yang berkaitan dengan kelas-kelas tarif terpopuler (V, X, Q). Sementara itu, pemasaran untuk kelas premium harus lebih tersegmentasi dan menonjolkan nilai lebih (fleksibilitas, layanan) daripada sekadar harga.
            """)
        class_popularity = df_cleaned['class'].value_counts().reset_index()
        class_popularity.columns = ['flight_class', 'count']
        fig6 = px.bar(class_popularity, x='flight_class', y='count')
        st.plotly_chart(fig6)
        with st.expander('Data Tabel', expanded=False):
            st.write("Data table with flight class popularity:")
            st.dataframe(class_popularity, hide_index=True)



    elif st.query_params["Section"] == "Q7: Apa saja 10 rute penerbangan terpopuler?":
        # Question 7: Top 10 most popular flight routes
        st.subheader(option)

        with st.expander('Hasil Analisis', expanded=False):
            st.markdown("""
            #### **Temuan Utama**

1.  **Dominasi Mutlak Jakarta (CGK) sebagai Super Hub:**
    * Fakta paling menonjol adalah bahwa **10 dari 10 rute teratas** melibatkan Bandara Internasional Soekarno-Hatta (CGK) Jakarta sebagai titik asal atau tujuan. Ini adalah bukti paling kuat yang mengukuhkan posisi Jakarta sebagai pusat (hub) utama dalam jaringan penerbangan nasional.

2.  **"Golden Triangle" dan Rute Utama (Trunk Routes):**
    * Rute-rute yang menghubungkan tiga kota besar: **Jakarta (CGK), Surabaya (SUB), dan Denpasar (DPS)**, membentuk "segitiga emas" lalu lintas udara di Indonesia. Rute CGK-SUB dan sebaliknya, serta CGK-DPS dan sebaliknya, mendominasi empat dari lima posisi teratas.
    * Selain itu, rute ke **Medan (KNO)** juga menunjukkan volume yang sangat tinggi, menjadikannya salah satu rute utama (trunk route) yang paling vital.

3.  **Keseimbangan Rute Dua Arah (Symmetrical Traffic):**
    * Untuk setiap pasangan kota, volume lalu lintasnya relatif seimbang di kedua arah. Contohnya:
        * CGK-SUB (82,357) vs SUB-CGK (77,992)
        * DPS-CGK (80,126) vs CGK-DPS (77,428)
        * PDG-CGK (50,643) vs CGK-PDG (49,036)
    * Ini menunjukkan aliran penumpang yang stabil dan berkelanjutan, baik untuk perjalanan bisnis, liburan, maupun keluarga di kedua arah, bukan lalu lintas satu arah yang bersifat musiman.
---
#### **Kesimpulan dan Implikasi Strategis**

* **Fokus pada Rute Utama:** Strategi bisnis harus memberikan prioritas tinggi pada rute-rute utama ini (CGK, SUB, DPS, KNO). Ketersediaan, harga yang kompetitif, dan frekuensi penerbangan di rute-rute ini adalah kunci untuk menguasai pangsa pasar terbesar.
* **Validasi Data Regional:** Analisis ini sangat mendukung temuan dari analisis regional (Q3) yang menunjukkan dominasi DKI Jakarta, serta pentingnya Jawa Timur (SUB), Bali (DPS), dan Sumatera Utara (KNO) sebagai pasar sekunder utama.
* **Peluang di Luar Hub:** Sementara rute-rute berbasis di Jakarta adalah yang paling dominan, ini juga membuka pertanyaan strategis: Adakah potensi yang belum tergali di rute-rute non-Jakarta (misalnya, SUB-DPS, SUB-UPG, atau DPS-LOP)? Mengembangkan rute titik-ke-titik (point-to-point) di luar Jakarta bisa menjadi strategi diferensiasi di masa depan.
            """)
        top_sectors = df_cleaned['sector'].value_counts().head(10).reset_index()
        top_sectors.columns = ['sector', 'frequency']
        fig7 = px.bar(top_sectors, x='sector', y='frequency')
        st.plotly_chart(fig7)
        with st.expander('Data Tabel', expanded=False):
            st.write("Data table with top flight routes:")
            st.dataframe(top_sectors, hide_index=True)



    elif st.query_params["Section"] == "Q8: Apa saja 10 maskapai dengan pendapatan tertinggi?":
        # Question 8: Top 10 airlines by revenue
        st.subheader(option)

        with st.expander('Hasil Analisis', expanded=False):
            st.markdown("""
            #### **Temuan Utama**

1.  **Pendapatan Tidak Selalu Sejalan dengan Volume:**
    * Fakta paling signifikan adalah bahwa peringkat pendapatan tidak sama persis dengan peringkat volume pesanan (dari Q1). Ini membuktikan bahwa jumlah penumpang yang banyak tidak secara otomatis menjamin pendapatan tertinggi.

2.  **Analisis Kompetitif LE vs. AZ:**
    * Meskipun memiliki jumlah pesanan yang hampir identik di Q1, **Maskapai LE (1095.3)** menghasilkan pendapatan yang jauh lebih tinggi daripada **Maskapai AZ (761.3)**.
    * Ini adalah wawasan kompetitif yang krusial: **Strategi LE lebih efektif dalam menghasilkan uang dari setiap penumpang.** Kemungkinan LE memiliki rata-rata harga tiket yang lebih tinggi, menjual lebih banyak layanan tambahan (ancillaries), atau fokus pada rute yang lebih menguntungkan dibandingkan AZ.

3.  **Dominasi XG Terkonfirmasi:**
    * Maskapai **XG** kokoh di posisi pertama baik dari segi volume maupun pendapatan, dengan total pendapatan **(1828.8)** yang hampir sama dengan gabungan pendapatan peringkat kedua dan ketiga. Ini menegaskan posisinya sebagai pemimpin pasar absolut.

4.  **Pemain Bernilai Tinggi (High-Value Player):**
    * Maskapai **WT** muncul di peringkat ke-8 dalam hal pendapatan **(207.5)**, meskipun tidak masuk dalam 10 besar dari segi volume pesanan. Ini menandakan WT adalah pemain "niche" yang efisien, mampu menghasilkan pendapatan signifikan dari jumlah penumpang yang relatif lebih sedikit, kemungkinan dengan menargetkan segmen premium atau rute bisnis.
---
#### **Kesimpulan dan Implikasi Strategis**

* **Fokus pada "Yield" bukan Hanya Volume:** Daripada hanya mengejar jumlah penumpang, strategi bisnis harus fokus pada peningkatan *yield* (pendapatan per penumpang). Kasus LE vs. AZ adalah contoh sempurna dari pentingnya strategi ini.
* **Pelajaran dari LE:** Perusahaan perlu menganalisis lebih dalam model bisnis LE. Apa yang mereka lakukan secara berbeda? Apakah dari sisi penetapan harga, manajemen rute, atau penawaran produk? Wawasan ini bisa menjadi cetak biru untuk meningkatkan profitabilitas.
* **Identifikasi Peluang Niche:** Keberhasilan pemain seperti WT menunjukkan adanya pasar yang menguntungkan di luar segmen massal. Mengidentifikasi dan melayani segmen premium ini bisa menjadi jalur pertumbuhan baru yang tidak memerlukan persaingan langsung dengan raksasa seperti XG.
            """)
        revenue_by_airline = df_cleaned.groupby('airline')['harga'].sum().reset_index()
        # Sort by average price from highest to lowest and get top 10
        revenue_by_airline = revenue_by_airline.sort_values('harga', ascending=False).head(10)
        fig8 = px.pie(revenue_by_airline, values='harga', names='airline')
        st.plotly_chart(fig8)
        with st.expander('Data Tabel', expanded=False):
            st.write("Data table with top airlines by revenue:")
            st.dataframe(revenue_by_airline, hide_index=True)



    elif st.query_params["Section"] == "Q9: Bagaimana perbedaan pola pemesanan berdasarkan hari dalam seminggu?":
        # Question 9: Booking patterns by day of the week
        st.subheader(option)

        with st.expander('Hasil Analisis', expanded=False):
            st.markdown("""
            #### **Temuan Utama**

1.  **Distribusi yang Sangat Merata:**
    * Temuan yang paling mengejutkan adalah betapa meratanya distribusi perjalanan sepanjang minggu. Perbedaan antara hari tersibuk **(Jumat, 533k)** dan hari tersepi **(Selasa, 502k)** hanya sekitar 6.3%.
    * Ini menunjukkan aliran penumpang yang sangat konsisten dan stabil, menandakan adanya perpaduan yang sehat antara **perjalanan bisnis (mid-week)** dan **perjalanan liburan (weekend-adjacent)**.

2.  **Puncak Perjalanan di Hari Jumat dan Minggu:**
    * **Jumat** adalah hari terpopuler untuk melakukan perjalanan, yang sangat sesuai dengan pola orang yang memulai liburan akhir pekan atau pulang ke kampung halaman.
    * **Minggu** adalah hari terpopuler ketiga, yang juga sangat logis karena merupakan hari utama bagi para pelancong untuk kembali ke kota asal mereka sebelum memulai minggu kerja baru.

3.  **Kekuatan Perjalanan Bisnis di Tengah Minggu:**
    * Hari **Rabu** dan **Kamis** menunjukkan volume perjalanan yang sangat tinggi, bahkan sedikit mengungguli hari Minggu. Ini adalah indikator kuat dari segmen perjalanan bisnis yang signifikan, yang sering melakukan perjalanan di pertengahan minggu.

4.  **Kontras dengan Pola Pembuatan Pesanan (Q5):**
    * Analisis ini menunjukkan kontras yang menarik dengan analisis tanggal pembuatan pesanan (Q5), di mana pesanan cenderung memuncak pada hari kerja dan turun drastis di akhir pekan.
    * Ini memperkuat hipotesis: **Pemesanan tiket adalah aktivitas "kerja" yang dilakukan pada hari kerja**, sedangkan **perjalanan itu sendiri terdistribusi lebih merata sepanjang minggu** untuk mengakomodasi baik pelancong bisnis maupun liburan.
---
#### **Kesimpulan dan Implikasi Strategis**

* **Strategi Harga yang Stabil:** Karena permintaan yang relatif merata, tidak ada justifikasi kuat untuk menaikkan harga secara drastis di akhir pekan. Strategi harga bisa dipertahankan agar relatif stabil, dengan kemungkinan penyesuaian kecil pada hari Jumat untuk menangkap permintaan puncak.
* **Peluang Pemasaran Tertarget:** Ada peluang untuk meluncurkan kampanye yang dirancang khusus untuk meningkatkan permintaan pada hari-hari dengan volume sedikit lebih rendah, seperti **Selasa** dan **Sabtu**. Contoh: "Diskon Terbang Selasa" atau "Bonus Liburan Sabtu".
* **Efisiensi Operasional:** Staf bandara dan maskapai dapat merencanakan jadwal mereka dengan asumsi bahwa volume penumpang akan tinggi dan konsisten sepanjang minggu, memungkinkan perencanaan sumber daya yang lebih efisien tanpa siklus "sibuk-sepi" yang ekstrem.
            """)
        df_clean_temp = df_cleaned.copy()
        # Check if travel_date is already datetime, if not convert it
        if not pd.api.types.is_datetime64_any_dtype(df_clean_temp['travel_date']):
            df_clean_temp['travel_date'] = pd.to_datetime(df_clean_temp['travel_date'])
        df_clean_temp['travel_weekday'] = df_clean_temp['travel_date'].dt.day_name()
        weekday_bookings = df_clean_temp['travel_weekday'].value_counts().reset_index()
        weekday_bookings.columns = ['weekday', 'bookings']
        # Reorder by weekday
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_bookings['weekday'] = pd.Categorical(weekday_bookings['weekday'], 
                                                   categories=weekday_order, ordered=True)
        weekday_bookings = weekday_bookings.sort_values('weekday')
        fig9 = px.bar(weekday_bookings, x='weekday', y='bookings')
        st.plotly_chart(fig9)
        with st.expander('Data Tabel', expanded=False):
            st.write("Data table with bookings by day of the week:")
            st.dataframe(weekday_bookings, hide_index=True)



    elif st.query_params["Section"] == "Ringkasan Analisis & Implikasi Strategis (Q1-Q9)":
        # Executive Summary & Strategic Implications
        st.subheader(option, divider="gray")
        st.markdown("""
**Gambaran Umum**
Analisis komprehensif terhadap data pemesanan penerbangan mengungkap sebuah model bisnis yang sangat terkonsentrasi, didorong oleh volume tinggi pada segmen harga ekonomis. Meskipun menunjukkan pertumbuhan yang kuat pada periode awal, terdapat anomali signifikan yang memerlukan perhatian strategis segera. Laporan ini merangkum tiga pilar strategis utama: struktur pasar, risiko ketergantungan, dan dinamika operasional.

### **1. Struktur Pasar: Model Volume Tinggi, Harga Ekonomis**
Data secara konsisten menunjukkan bahwa mesin penggerak bisnis adalah penjualan tiket dengan volume tinggi pada harga yang sensitif.
* **Dominasi Kelas Ekonomi Promo:** Sebagian besar pemesanan berasal dari kelas tarif terendah (V, X, Q), mengonfirmasi bahwa mayoritas pelanggan memprioritaskan harga.
* **Pemain Utama Berbasis Volume:** Maskapai dengan volume penumpang terbesar (XG, LE, AZ) juga merupakan kontributor pendapatan terbesar, menegaskan model bisnis ini.
* **Jantung Operasi di Jakarta:** Seluruh rute terpopuler terhubung dengan Jakarta (CGK), menjadikannya "super hub" yang tak tergantikan dalam jaringan ini.

### **2. Risiko & Peluang: Ketergantungan vs. Diversifikasi**
Struktur pasar yang terkonsentrasi menciptakan efisiensi, namun juga menyimpan risiko signifikan.
* **Ancaman Ketergantungan Ganda:** Bisnis ini sangat bergantung pada satu maskapai (XG) dan satu hub (CGK). Setiap gangguan pada keduanya akan berdampak masif pada keseluruhan operasional.
* **Studi Kasus Profitabilitas (LE vs. AZ):** Meskipun volume penumpangnya hampir sama, Maskapai LE menghasilkan pendapatan yang jauh lebih tinggi daripada AZ. Ini membuktikan bahwa strategi monetisasi LE lebih efektif dan menjadi pelajaran penting: **fokus harus pada *yield* (pendapatan per penumpang), bukan hanya volume.**
* **Peluang Diversifikasi:** Pasar sekunder utama seperti Surabaya (SUB), Denpasar (DPS), dan Medan (KNO) menunjukkan volume yang kuat. Mengembangkan rute titik-ke-titik (point-to-point) yang tidak melibatkan Jakarta dapat menjadi strategi diferensiasi dan mitigasi risiko.

### **3. Dinamika Operasional & Anomali Kritis**
Pola pemesanan dan perjalanan memberikan wawasan mendalam tentang perilaku pelanggan dan satu anomali besar.
* **Ritme Bisnis Mingguan:** Pemesanan tiket adalah aktivitas hari kerja (Senin-Jumat), sering kali didorong oleh promo. Namun, perjalanan itu sendiri terdistribusi secara merata sepanjang minggu. Ini memisahkan "aksi jual" (hari kerja) dari "aksi bepergian" (sepanjang minggu).
* **Musim Puncak Teridentifikasi:** Permintaan perjalanan memuncak selama periode libur Lebaran dan Natal-Tahun Baru, memberikan pola yang dapat diprediksi untuk perencanaan.
* **INVESTIGASI KRITIS - Penurunan Drastis Awal 2019:** Data menunjukkan volume bisnis, baik pemesanan maupun perjalanan, jatuh secara drastis setelah Januari 2019. Ini adalah **temuan paling krusial dan berisiko tinggi**. Tanpa memahami penyebabnya (perubahan bisnis, faktor eksternal, atau kesalahan data), semua perencanaan strategis dan peramalan menjadi tidak valid.
---
### **Implikasi & Rekomendasi Strategis Utama**
1.  **Prioritas #1: Investigasi Anomali 2019.** Selidiki segera penyebab penurunan volume setelah Januari 2019. Jawaban atas pertanyaan ini akan menentukan arah strategi ke depan.
2.  **Geser Fokus dari Volume ke Nilai (Yield).** Pelajari model bisnis Maskapai LE. Implementasikan strategi untuk meningkatkan pendapatan per penumpang, misalnya melalui layanan tambahan (ancillaries) atau manajemen harga yang lebih dinamis.
3.  **Buat Rencana Diversifikasi Rute.** Kurangi ketergantungan pada Jakarta dengan secara aktif mengembangkan rute antar-hub sekunder (misal: SUB-UPG, DPS-LOP) untuk membangun keunggulan kompetitif baru.
4.  **Optimalkan Pemasaran & Operasional.** Luncurkan kampanye promosi besar pada hari kerja. Pertahankan staf operasional pada tingkat yang konsisten sepanjang minggu, dengan antisipasi puncak pada hari Jumat.
""")