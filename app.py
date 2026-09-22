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
# 4. HALAMAN LOGIN (TEMA DARK INDUSTRIAL)
# ==========================================
def show_login_page():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800;900&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        
        #MainMenu, header, footer {visibility: hidden;}
        
        /* 1. BACKGROUND GELAP INDUSTRIAL */
        .stApp {
            background-color: #121212 !important;
            background-image: 
                radial-gradient(#2A2A2A 1px, transparent 1px) !important;
            background-position: 0 0 !important;
            background-size: 30px 30px !important;
        }
        
        .block-container { 
            padding-top: 10vh !important; 
            max-width: 1100px !important; 
            margin: 0 auto !important;
        }
        
        /* 2. PANEL LOGIN DENGAN NEON GLOW LEBIH TERANG */
        [data-testid="column"]:nth-of-type(1) {
            background: rgba(20, 20, 25, 0.8) !important;
            backdrop-filter: blur(15px) !important;
            -webkit-backdrop-filter: blur(15px) !important;
            border-radius: 16px !important;
            padding: 45px 40px !important;
            border: 2px solid rgba(66, 133, 244, 0.6) !important; /* Border neon lebih tebal & terang */
            box-shadow: 0 0 20px rgba(66, 133, 244, 0.5), 0 0 45px rgba(66, 133, 244, 0.3), inset 0 0 20px rgba(0,0,0,0.8) !important; /* Pendaran Neon Berlapis */
            margin-top: 20px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        /* 3. PANEL KANAN (GAMBAR 3D) */
        [data-testid="column"]:nth-of-type(2) {
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important; 
            align-items: flex-start !important; 
            padding-left: 20px !important;
            margin-top: 20px;
        }
        
        /* Styling Input Field Mode Gelap */
        .stTextInput>div>div>input { 
            border: 1px solid #444444 !important; 
            border-radius: 8px !important; 
            padding: 14px 15px !important; 
            background-color: #1A1A1A !important;
            color: #FFFFFF !important;
            font-weight: 500 !important;
            transition: all 0.3s ease;
        }
        
        /* Glow Biru Gemini saat diklik */
        .stTextInput>div>div>input:focus {
            border-color: #4285F4 !important;
            box-shadow: 0 0 15px rgba(66, 133, 244, 0.6) !important;
        }
        
        /* Teks Label Input */
        .stTextInput label p {
            color: #A0A0A0 !important;
        }
        
        /* Tombol Industrial Glow - Biru Gemini */
        .stButton>button {
            background: linear-gradient(135deg, #1A73E8 0%, #0D47A1 100%) !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            border: 1px solid #4285F4 !important;
            font-weight: 800 !important;
            font-size: 16px !important;
            padding: 12px 24px !important;
            box-shadow: 0 0 25px rgba(26, 115, 232, 0.7) !important; /* Neon glow tombol lebih terang */
            transition: all 0.3s ease;
            margin-top: 15px;
            letter-spacing: 1px;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #4285F4 0%, #1565C0 100%) !important;
            box-shadow: 0 0 40px rgba(66, 133, 244, 0.9) !important;
            transform: translateY(-2px);
        }
        </style>
    """, unsafe_allow_html=True)

    col_kiri, col_kanan = st.columns([1, 1.1], gap="small")

    with col_kiri:
        st.markdown("""
            <div style="
                background: linear-gradient(145deg, #222222, #1c1c1c);
                border-radius: 12px;
                padding: 12px 20px;
                box-shadow: 6px 6px 12px #111111, -6px -6px 12px #2b2b2b, 0 0 20px rgba(66, 133, 244, 0.4);
                border: 1px solid rgba(66, 133, 244, 0.5);
                display: inline-block;
                margin-bottom: 25px;
            ">
                <h3 style='margin:0; font-weight: 900; font-size: 26px; color: #E0E0E0; letter-spacing: 1px;'>
                    <span style="color:#4285F4; text-shadow: 0 0 12px rgba(66, 133, 244, 0.8);">⚙️</span> Cementify
                </h3>
            </div>
            <h1 style='font-size: 42px; color: #FFFFFF; font-weight: 800; margin-bottom: 5px; line-height: 1.2;'>Welcome Back!</h1>
            <p style='color: #888888; margin-bottom: 35px; font-weight: 500;'>Please login to manage your inventory</p>
        """, unsafe_allow_html=True)

        username = st.text_input("Username / Email", placeholder="admin")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        st.markdown("<div style='text-align: right; color: #888888; font-size: 13px; margin-top: -10px; margin-bottom: 20px; cursor: pointer; font-weight: 600; transition: color 0.3s;' onmouseover=\"this.style.color='#4285F4'\" onmouseout=\"this.style.color='#888888'\">Forgot Password?</div>", unsafe_allow_html=True)

        if st.button("LOGIN", use_container_width=True):
            if username.lower() == "admin" and password == "indocement123":
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("Kredensial tidak valid.")

    with col_kanan:
        img_path = "gambar ilustrasi web.jpg" 
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                img_data = f.read()
            img_b64 = base64.b64encode(img_data).decode()
            
            st.markdown(f"""
                <div style="
                    width: 100%;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: flex-start;
                ">
                    <div style="
                        width: 100%;
                        max-width: 580px;
                        height: 420px; /* KUNCI: Tinggi fix agar tidak melar jadi persegi panjang vertikal */
                        background: #1E1E1E;
                        border-radius: 20px;
                        border: 2px solid rgba(66, 133, 244, 0.6); /* Border neon lebih terang */
                        box-shadow: 0 0 20px rgba(66, 133, 244, 0.5), 0 0 45px rgba(66, 133, 244, 0.3), 0 15px 35px rgba(0, 0, 0, 0.6); /* Pendaran Neon Berlapis */
                        overflow: hidden;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    ">
                        <img src="data:image/jpeg;base64,{img_b64}" style="
                            width: 100%; 
                            height: 100%;
                            object-fit: cover;
                            object-position: center;
                            display: block;
                        "/>
                    </div>
                    <div style="width: 100%; max-width: 580px; text-align: right; color: #555555; font-size: 13px; margin-top: 15px; font-weight: 600;">
                        Support &nbsp;&nbsp;&nbsp; Terms
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Foto tidak ditemukan.")

# ==========================================
# 5. HALAMAN DASHBOARD UTAMA
# ==========================================
def show_dashboard_page():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        
        /* Background Dashboard Biru Muda Lembut */
        .stApp { background: #EAF0F8 !important; }
        
        [data-testid="stSidebar"] {
            background-color: #1A233A !important;
        }
        [data-testid="stSidebar"] * {
            color: #FFFFFF !important;
        }
        
        /* Perbaikan Tombol Log Out agar teksnya terlihat */
        .stSidebar .stButton > button {
            background-color: transparent !important;
            color: #FFFFFF !important;
            border: 1px solid #FFFFFF !important;
        }
        .stSidebar .stButton > button:hover {
            background-color: rgba(255,255,255,0.1) !important;
        }
        
        /* Mengembalikan padding normal untuk Dashboard */
        .block-container { 
            padding: 2rem 4rem !important; 
            max-width: 100% !important; 
        }
        
        .stTabs [data-baseweb="tab-list"] { 
            gap: 10px; 
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #FFFFFF;
            color: #1E293B !important;
            border-radius: 8px 8px 0px 0px;
            padding: 10px 20px;
            font-weight: 600;
            box-shadow: 0 -2px 4px rgba(0,0,0,0.02);
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #159895 0%, #2B3A5A 100%) !important;
            color: #FFFFFF !important;
        }
        
        [data-testid="stFileUploadDropzone"] {
            background: linear-gradient(135deg, rgba(43, 58, 90, 0.85) 0%, rgba(21, 152, 149, 0.85) 100%) !important;
            border: 2px dashed #94A3B8 !important;
            border-radius: 12px !important;
            padding: 30px !important;
        }
        [data-testid="stFileUploadDropzone"] * {
            color: #FFFFFF !important;
        }
        [data-testid="stFileUploadDropzone"] button {
            background-color: #FFFFFF !important;
            color: #1A233A !important;
            border-radius: 20px !important;
            font-weight: bold !important;
        }
        
        .btn-proses>button {
            background: linear-gradient(90deg, #1A233A 0%, #2B3A5A 100%) !important; 
            color: #FFFFFF !important; 
            font-weight: 800 !important;
            border-radius: 8px !important; 
            width: 100% !important; 
            border: 1px solid #159895 !important;
            padding: 15px !important;
            margin-top: 15px;
            box-shadow: 0 4px 15px rgba(21, 152, 149, 0.3) !important;
        }
        
        div.stInfo {
            background: linear-gradient(90deg, #E0F2F1 0%, #B2DFDB 100%) !important;
            color: #004D40 !important;
            border-left: 5px solid #00897B !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 10px rgba(0, 137, 123, 0.1) !important;
        }
        div.stSuccess {
            background: linear-gradient(90deg, #E8F5E9 0%, #C8E6C9 100%) !important;
            color: #1B5E20 !important;
            border-left: 5px solid #4CAF50 !important;
        }
        
        [data-testid="stDataFrame"] {
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            overflow: hidden;
            background-color: #FFFFFF;
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
        
        # Banner Gambar
        banner_path = "gambar menu.jpg"
        if os.path.exists(banner_path):
             with open(banner_path, "rb") as f:
                img_data = f.read()
             img_b64 = base64.b64encode(img_data).decode()
             st.markdown(f"""
                <div style="
                    width: 100%;
                    height: 250px;
                    border-radius: 12px;
                    background-image: url('data:image/jpeg;base64,{img_b64}');
                    background-size: cover;
                    background-position: center;
                    margin-bottom: 25px;
                    box-shadow: 0 8px 16px rgba(0,0,0,0.15);
                "></div>
             """, unsafe_allow_html=True)
        else:
             st.warning(f"Foto banner '{banner_path}' tidak ditemukan.")

        st.markdown("<h3 style='color: #1E293B; margin-top: -10px;'>📸 Capture & Upload</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #64748B;'>Gunakan panel di bawah untuk mendeteksi tumpukan semen.</p>", unsafe_allow_html=True)
        
        col_kiri, col_kanan = st.columns([1.2, 1], gap="large")
        
        with col_kiri:
            tab1, tab2 = st.tabs(["📁 Unggah Berkas", "📷 Ambil Foto via Kamera"])
            
            with tab1:
                st.markdown("**Area Unggah Gambar (Drag & Drop)**")
                active_source_1 = st.file_uploader("", type=['jpg', 'jpeg', 'png', 'webp'], key="up1", label_visibility="collapsed")
            with tab2:
                st.markdown("**Gunakan Kamera Perangkat**")
                active_source_2 = st.camera_input("", key="cam1", label_visibility="collapsed")
            
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
            st.markdown("<h4 style='color: #1E293B;'>🔍 Detection Results</h4>", unsafe_allow_html=True)
            if st.session_state.get('latest_scan') is not None:
                st.image(st.session_state['latest_scan'], use_container_width=True)
                st.success(f"**{st.session_state['latest_info']}**")
                if st.session_state.get('latest_details'):
                    st.caption(f"Rincian Merek: {st.session_state['latest_details']}")
            else:
                st.info("⏳ Menunggu masukan gambar dari operator.")

        st.markdown("---")
        st.markdown("<h4 style='color: #159895;'>📋 Recent Detections (Live Table)</h4>", unsafe_allow_html=True)
        if not st.session_state["history_table"].empty:
            st.dataframe(st.session_state["history_table"].tail(5), hide_index=True, use_container_width=True)
        else:
            st.caption("Data pemindaian akan muncul di sini secara otomatis.")

    elif menu == "📁 Log Lengkap (CSV)":
        st.markdown("<h2 style='color: #1E293B;'>Database Riwayat Inventori</h2>", unsafe_allow_html=True)
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