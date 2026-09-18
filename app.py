import streamlit as st
import pandas as pd
import datetime
import io
import base64
import requests
import cv2
import os
from PIL import Image
from ultralytics import YOLO

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA
# ==========================================
st.set_page_config(page_title="Cementify | Smart Inventory", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 2. INISIALISASI MEMORI (SESSION STATE)
# ==========================================
def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if "history_table" not in st.session_state:
        st.session_state["history_table"] = pd.DataFrame(columns=["Tanggal", "Waktu", "Total Sak", "Rincian Merek", "Tautan Foto"])
    if "latest_scan" not in st.session_state:
        st.session_state["latest_scan"] = None
    if "latest_info" not in st.session_state:
        st.session_state["latest_info"] = None
    if "latest_details" not in st.session_state:
        st.session_state["latest_details"] = None

init_session_state()

# ==========================================
# 3. FUNGSI BACKEND: API GAMBAR & AI YOLOv8
# ==========================================
def upload_to_imgbb(image_pil):
    try:
        buffered = io.BytesIO()
        image_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        api_key = "43412f31ad19391f1fa288300a199105" 
        url = "https://api.imgbb.com/1/upload"
        payload = {"key": api_key, "image": img_str}
        res = requests.post(url, payload, timeout=15)
        
        if res.status_code == 200:
            return res.json()['data']['url_viewer']
        return "Gagal Upload Server"
    except Exception:
        return "Error Jaringan"

@st.cache_resource
def load_vision_model():
    try:
        return YOLO('best.pt')
    except Exception as e:
        return None

def process_image(image_file, model):
    try:
        img = Image.open(image_file)
        results = model.predict(source=img, conf=0.5, iou=0.7, agnostic_nms=True)
        detection = results[0]
        
        img_array = detection.plot()
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        processed_img = Image.fromarray(img_rgb)
        
        class_names = model.names
        inventory_counts = {}
        
        for box in detection.boxes.cls:
            brand_name = class_names[int(box)]
            inventory_counts[brand_name] = inventory_counts.get(brand_name, 0) + 1
            
        total_detected = len(detection.boxes.cls)
        brand_details = ", ".join([f"{k}: {v}" for k, v in inventory_counts.items()]) if inventory_counts else "Kosong"
        
        return processed_img, total_detected, brand_details
    except Exception as e:
        return None, 0, "Error"

# ==========================================
# 4. HALAMAN LOGIN (DESAIN SPLIT KONTRAK PRESISI)
# ==========================================
def show_login_page():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        
        #MainMenu, header, footer {visibility: hidden;}
        
        /* Membelah warna latar belakang (50% Putih Abu, 50% Biru Gelap) SEIMBANG */
        .stApp {
            background: linear-gradient(to right, #F8F9FA 50%, #2B3A5A 50%) !important;
        }
        
        /* Margin aman Kiri & Kanan dikurangi sedikit agar area lebih luas */
        .block-container { 
            padding-top: 15vh !important; 
            padding-bottom: 0 !important;
            padding-left: 5vw !important;  
            padding-right: 5vw !important; 
            max-width: 100% !important; 
        }
        
        /* Desain Kolom Input dengan Garis Batas (Border) Tegas */
        .stTextInput>div>div>input { 
            border: 1px solid #94A3B8 !important; 
            border-radius: 8px !important; 
            padding: 12px 15px !important; 
            background-color: #FFFFFF !important;
            color: #0F172A !important;
        }
        
        /* Desain Tombol LOGIN Biru Gelap */
        .stButton>button {
            background-color: #2B3A5A !important;
            color: #FFFFFF !important;
            border-radius: 30px !important;
            border: none !important;
            font-weight: 700 !important;
            padding: 10px 24px !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #1A233A !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Proporsi 1:1 agar formulir dan gambar berbagi ruang 50/50 secara adil
    col_kiri, col_kanan = st.columns([1, 1], gap="large")

    with col_kiri:
        st.markdown("<div style='padding-right: 12%;'>", unsafe_allow_html=True)
        st.markdown("""
            <h3 style='margin:0; font-weight: 800; color: #1E293B; font-size: 24px; margin-bottom: 25px;'>Cementify ⚙️</h3>
            <h1 style='font-size: 40px; color: #0F172A; font-weight: 800; margin-bottom: 5px;'>Welcome Back!</h1>
            <p style='color: #64748B; margin-bottom: 35px;'>Please login to manage your inventory</p>
        """, unsafe_allow_html=True)

        username = st.text_input("Username / Email", placeholder="admin")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        st.markdown("<div style='text-align: right; color: #64748B; font-size: 14px; margin-top: -10px; margin-bottom: 25px; cursor: pointer;'>Forgot Password?</div>", unsafe_allow_html=True)

        if st.button("LOGIN", use_container_width=True):
            if username.lower() == "admin" and password == "indocement123":
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("Kredensial tidak valid. Silakan periksa kembali.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_kanan:
        img_path = "gambar ilustrasi web.jpg"
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                img_data = f.read()
            img_b64 = base64.b64encode(img_data).decode()
            
            # Trik melebarkan gambar: Hapus padding kanan (menjadi 0)
            st.markdown(f"""
                <div style="
                    display: flex; 
                    flex-direction: column;
                    justify-content: flex-start; 
                    align-items: center; 
                    padding: 0 0 0 4vw; /* Atas 0, Kanan 0 (Melebar), Bawah 0, Kiri 4vw (Menjaga Jarak dari Tengah) */
                ">
                    <img src="data:image/jpeg;base64,{img_b64}" style="
                        width: 100%; 
                        height: 65vh;
                        object-fit: cover;
                        border-radius: 20px; 
                        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
                    "/>
                    <div style="width: 100%; text-align: right; color: #94A3B8; font-size: 12px; margin-top: 20px;">
                        Support &nbsp;&nbsp;&nbsp; Terms
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Foto 'gambar ilustrasi web.jpg' tidak ditemukan di GitHub.")

# ==========================================
# 5. HALAMAN DASHBOARD UTAMA
# ==========================================
def show_dashboard_page():
    st.markdown("""
        <style>
        /* Mengembalikan Background Dashboard menjadi Putih Bersih */
        .stApp { background: #FFFFFF !important; }
        
        [data-testid="stSidebar"] {
            background-color: #1A233A !important;
        }
        [data-testid="stSidebar"] * {
            color: #FFFFFF !important;
        }
        
        .btn-proses>button {
            background-color: #F5A623 !important; 
            color: #1A233A !important; 
            font-weight: 800 !important;
            border-radius: 8px !important; 
            width: 100% !important; 
            border: none !important;
            padding: 12px !important;
            margin-top: 10px;
        }
        
        /* Reset padding konten untuk Dashboard agar kembali normal */
        .block-container { 
            padding: 3rem 4rem !important; 
            max-width: 100% !important; 
        }
        
        .stTabs [data-baseweb="tab-list"] { gap: 20px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #f8f9fa;
            border-radius: 8px 8px 0px 0px;
            padding: 10px 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 🏢 Cementify")
        st.markdown("---")
        menu = st.radio("Navigasi", ["📊 Dashboard Utama", "📁 Log Lengkap (CSV)"])
        st.markdown("---")
        if st.button("🚪 Keluar (Log Out)", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()

    model = load_vision_model()

    if menu == "📊 Dashboard Utama":
        st.markdown("### 📸 Capture & Upload")
        st.markdown("Gunakan panel di bawah untuk mendeteksi tumpukan semen.")
        
        col_kiri, col_kanan = st.columns([1.2, 1], gap="large")
        
        with col_kiri:
            tab1, tab2 = st.tabs(["📁 Unggah Berkas", "📷 Ambil Foto via Kamera"])
            
            with tab1:
                st.markdown("**Area Unggah Gambar (Drag & Drop)**")
                active_source_1 = st.file_uploader("", type=['jpg', 'jpeg', 'png', 'webp'], key="up1")
            with tab2:
                st.markdown("**Gunakan Kamera Perangkat**")
                active_source_2 = st.camera_input("", key="cam1")
            
            active_source = active_source_1 if active_source_1 else active_source_2
            
            st.markdown('<div class="btn-proses">', unsafe_allow_html=True)
            if st.button("🚀 PROSES GAMBAR SEKARANG"):
                if not active_source:
                    st.warning("Silakan masukkan gambar terlebih dahulu.")
                elif not model:
                    st.error("Model AI (best.pt) tidak ditemukan. Pastikan sudah diunggah.")
                else:
                    with st.spinner("YOLOv8 sedang menganalisis gambar..."):
                        result_img, total_sacks, details = process_image(active_source, model)
                        
                        if result_img:
                            now = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
                            date_str = now.strftime("%Y-%m-%d")
                            time_str = now.strftime("%H:%M:%S")
                            
                            st.toast("Menyimpan foto ke Cloud ImgBB...", icon="☁️")
                            img_url = upload_to_imgbb(result_img)
                            
                            st.session_state['latest_scan'] = result_img
                            st.session_state['latest_info'] = f"Total Terdeteksi: {total_sacks} Sak"
                            st.session_state['latest_details'] = details
                            
                            new_row = pd.DataFrame([{
                                "Tanggal": date_str, 
                                "Waktu": time_str, 
                                "Total Sak": total_sacks, 
                                "Rincian Merek": details, 
                                "Tautan Foto": img_url
                            }])
                            st.session_state["history_table"] = pd.concat([st.session_state["history_table"], new_row], ignore_index=True)
                            
                            st.success("Pemindaian Berhasil!")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_kanan:
            st.markdown("#### 🔍 Detection Results")
            if st.session_state.get('latest_scan') is not None:
                st.image(st.session_state['latest_scan'], use_container_width=True)
                st.success(f"**{st.session_state['latest_info']}**")
                if st.session_state.get('latest_details'):
                    st.caption(f"Rincian Merek: {st.session_state['latest_details']}")
            else:
                st.info("Menunggu masukan gambar dari operator.")

        st.markdown("---")
        st.markdown("#### 📋 Recent Detections (Live Table)")
        if not st.session_state["history_table"].empty:
            st.dataframe(st.session_state["history_table"].tail(5), hide_index=True, use_container_width=True)
        else:
            st.caption("Data pemindaian akan muncul di sini secara otomatis.")

    elif menu == "📁 Log Lengkap (CSV)":
        st.title("Database Riwayat Inventori")
        st.markdown("Semua data pemindaian selama aplikasi berjalan direkam di sini.")
        
        st.dataframe(st.session_state["history_table"], hide_index=True, use_container_width=True)
        
        if not st.session_state["history_table"].empty:
            st.markdown("<br>", unsafe_allow_html=True)
            csv = st.session_state["history_table"].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Unduh Data ke Excel (CSV)",
                data=csv,
                file_name=f'Database_Semen_{datetime.datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
                use_container_width=True,
                type="primary"
            )
        else:
            st.info("Belum ada data pemindaian pada sesi ini.")

# ==========================================
# 6. ROUTING UTAMA
# ==========================================
if __name__ == "__main__":
    if not st.session_state['logged_in']:
        show_login_page()
    else:
        show_dashboard_page()