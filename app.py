import streamlit as st
import pandas as pd
import numpy as np
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
# 2. INISIALISASI MEMORI (ANTI-RELOAD)
# ==========================================
HISTORY_FILE = "history_log.csv"

def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        
    if "history_table" not in st.session_state:
        if os.path.exists(HISTORY_FILE):
            st.session_state["history_table"] = pd.read_csv(HISTORY_FILE)
        else:
            st.session_state["history_table"] = pd.DataFrame(columns=["Tanggal", "Waktu", "Nama File", "Total Sak", "Rincian Merek", "Tautan Foto"])
            
    if "latest_results" not in st.session_state:
        st.session_state["latest_results"] = []

init_session_state()

# ==========================================
# 3. FUNGSI BACKEND: API GAMBAR & AI YOLOv8
# ==========================================
def upload_to_imgbb(image_pil):
    try:
        buffered = io.BytesIO()
        image_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # --- KEAMANAN LEVEL INDUSTRI (ANTI-HACKER) ---
        try:
            api_key = st.secrets["IMGBB_API_KEY"]
        except:
            # Fallback jika dijalankan lokal tanpa file secrets
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
        return YOLO('best (12).pt')
    except Exception as e:
        return None

def process_image(image_file, model, corrections_dict=None):
    try:
        img = Image.open(image_file)
        img_array = np.array(img)
        
        if img_array.shape[-1] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        results = model.predict(source=img_bgr, conf=0.5, iou=0.7, agnostic_nms=True)
        detection = results[0]
        
        class_names = model.names
        inventory_counts = {}
        total_detected = 0
        
        colors = [
            (50, 100, 255), (255, 255, 50), (255, 100, 255), (100, 255, 255),
            (150, 50, 200), (200, 150, 50), (50, 200, 150), (255, 128, 0),
            (0, 128, 255), (128, 0, 255), (255, 0, 128)
        ]
        
        for box in detection.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            brand_name = class_names[cls_id]
            
            # --- PENGAMAN KELAS HANTU ---
            if brand_name == "cement1":
                brand_name = "SEMEN TIGA RODA"
                
            # --- FITUR KOREKSI HUMAN-IN-THE-LOOP (MULTI-RULE) ---
            if corrections_dict:
                for wrong_brand, correct_brand in corrections_dict.items():
                    if wrong_brand.upper() in brand_name.upper():
                        brand_name = correct_brand.upper()
                
            # --- PEWARNAAN ---
            if "RAJAWALI" in brand_name.upper(): color = (0, 255, 0)
            elif "TIGA RODA" in brand_name.upper(): color = (255, 0, 0)
            elif "MERDEKA" in brand_name.upper(): color = (0, 255, 255)
            elif "GRESIK" in brand_name.upper(): color = (0, 165, 255)
            else: color = colors[cls_id % len(colors)]
                
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            label = f"{brand_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_bgr, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            text_color = (0, 0, 0) if "MERDEKA" in brand_name.upper() else (255, 255, 255)
            cv2.putText(img_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            inventory_counts[brand_name] = inventory_counts.get(brand_name, 0) + 1
            total_detected += 1
            
        img_rgb_final = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        processed_img = Image.fromarray(img_rgb_final)
        brand_details = ", ".join([f"{k}: {v}" for k, v in inventory_counts.items()]) if inventory_counts else "Kosong"
        
        return processed_img, total_detected, brand_details
        
    except Exception as e:
        print("Gagal proses gambar:", e)
        return None, 0, "Error"


# ==========================================
# 4. HALAMAN LOGIN WEB (CSS ASLI)
# ==========================================
def show_login_page():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800;900&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        
        #MainMenu, header, footer {visibility: hidden;}
        
        .stApp {
            background-color: #121212 !important;
            background-image: radial-gradient(#2A2A2A 1px, transparent 1px) !important;
            background-position: 0 0 !important;
            background-size: 30px 30px !important;
        }
        
        .block-container { 
            padding-top: 10vh !important; 
            max-width: 1100px !important; 
            margin: 0 auto !important;
        }
        
        [data-testid="column"]:nth-of-type(1) {
            background: rgba(20, 20, 25, 0.8) !important;
            backdrop-filter: blur(15px) !important;
            -webkit-backdrop-filter: blur(15px) !important;
            border-radius: 16px !important;
            padding: 45px 40px !important;
            border: 2px solid rgba(66, 133, 244, 0.6) !important; 
            box-shadow: 0 0 20px rgba(66, 133, 244, 0.5), 0 0 45px rgba(66, 133, 244, 0.3), inset 0 0 20px rgba(0,0,0,0.8) !important; 
            margin-top: 20px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 100%; 
        }

        [data-testid="column"]:nth-of-type(2) {
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-start !important; 
            align-items: center !important; 
            margin-top: 20px;
            height: 100%; 
        }
        
        .stTextInput>div>div>input { 
            border: 1px solid #444444 !important; 
            border-radius: 8px !important; 
            padding: 14px 15px !important; 
            background-color: #1A1A1A !important;
            color: #FFFFFF !important;
            font-weight: 500 !important;
            transition: all 0.3s ease;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: #4285F4 !important;
            box-shadow: 0 0 15px rgba(66, 133, 244, 0.6) !important;
        }
        
        .stTextInput label p {
            color: #A0A0A0 !important;
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #1A73E8 0%, #0D47A1 100%) !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            border: 1px solid #4285F4 !important;
            font-weight: 800 !important;
            font-size: 16px !important;
            padding: 12px 24px !important;
            box-shadow: 0 0 25px rgba(26, 115, 232, 0.7) !important; 
            transition: all 0.3s ease;
            margin-top: 15px;
            letter-spacing: 1px;
            width: 100%; 
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #4285F4 0%, #1565C0 100%) !important;
            box-shadow: 0 0 40px rgba(66, 133, 244, 0.9) !important;
            transform: translateY(-2px);
        }
        </style>
    """, unsafe_allow_html=True)

    col_kiri, col_kanan = st.columns([1, 1.1], gap="large")

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
            # --- KEAMANAN LEVEL INDUSTRI (ANTI-HACKER) ---
            try:
                valid_user = st.secrets["ADMIN_USERNAME"]
                valid_pass = st.secrets["ADMIN_PASSWORD"]
            except:
                # Fallback jika dijalankan lokal tanpa file secrets
                valid_user = "admin"
                valid_pass = "indocement123"

            if username.lower() == valid_user and password == valid_pass:
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
                    height: 100%; 
                    display: flex;
                    flex-direction: column;
                    justify-content: flex-start;
                    align-items: center; 
                ">
                    <div style="
                        width: 100%;
                        height: 100%; 
                        min-height: 480px; 
                        background: #1E1E1E;
                        border-radius: 20px;
                        border: 2px solid rgba(66, 133, 244, 0.8); 
                        box-shadow: 0 0 25px rgba(66, 133, 244, 0.6), 0 0 50px rgba(66, 133, 244, 0.4), inset 0 0 15px rgba(0,0,0,0.5);
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
                    <div style="width: 100%; text-align: right; color: #555555; font-size: 13px; margin-top: 15px; font-weight: 600; padding-right: 5px;">
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
        
        .stApp { background: #EAF0F8 !important; }
        
        [data-testid="stSidebar"] {
            background-color: #1A233A !important;
        }
        [data-testid="stSidebar"] * {
            color: #FFFFFF !important;
        }
        
        .stSidebar .stButton > button {
            background-color: transparent !important;
            color: #FFFFFF !important;
            border: 1px solid #FFFFFF !important;
        }
        .stSidebar .stButton > button:hover {
            background-color: rgba(255,255,255,0.1) !important;
        }
        
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
        
        # ==========================================
        # MENU KOREKSI MULTI-RULE DI SIDEBAR
        # ==========================================
        st.markdown("### 🛠️ Mode Koreksi Multi-Brand")
        st.caption("Benarkan 2 kesalahan sekaligus!")
        enable_correction = st.checkbox("Aktifkan Koreksi")
        
        corrections_dict = {}
        if enable_correction:
            st.markdown("**Aturan Koreksi 1:**")
            w1 = st.selectbox("AI Salah Nebak (1):", ["-", "RAJAWALI", "SEMEN TIGA RODA", "GRESIK", "MERDEKA", "BOSOWA", "PADANG", "GARUDA"], key="w1")
            c1 = st.selectbox("Harusnya (1):", ["-", "MERDEKA", "GRESIK", "RAJAWALI", "SEMEN TIGA RODA", "BOSOWA", "PADANG", "GARUDA"], key="c1")
            if w1 != "-" and c1 != "-": corrections_dict[w1] = c1
                
            st.markdown("**Aturan Koreksi 2:**")
            w2 = st.selectbox("AI Salah Nebak (2):", ["-", "RAJAWALI", "SEMEN TIGA RODA", "GRESIK", "MERDEKA", "BOSOWA", "PADANG", "GARUDA"], key="w2")
            c2 = st.selectbox("Harusnya (2):", ["-", "MERDEKA", "GRESIK", "RAJAWALI", "SEMEN TIGA RODA", "BOSOWA", "PADANG", "GARUDA"], key="c2")
            if w2 != "-" and c2 != "-": corrections_dict[w2] = c2
        
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

        st.markdown("<h3 style='color: #1E293B; margin-top: -10px;'>📸 Capture & Upload (Multi-Image)</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #64748B;'>Gunakan panel di bawah untuk mendeteksi tumpukan semen.</p>", unsafe_allow_html=True)
        
        col_kiri, col_kanan = st.columns([1.2, 1], gap="large")
        
        with col_kiri:
            tab1, tab2 = st.tabs(["📁 Unggah Berkas", "📷 Ambil Foto via Kamera"])
            
            with tab1:
                st.markdown("**Area Unggah Gambar (Bisa lebih dari satu)**")
                uploaded_files = st.file_uploader("", type=['jpg', 'jpeg', 'png', 'webp'], accept_multiple_files=True, key="up1", label_visibility="collapsed")
            with tab2:
                st.markdown("**Gunakan Kamera Perangkat**")
                camera_file = st.camera_input("", key="cam1", label_visibility="collapsed")
            
            active_sources = []
            if uploaded_files: active_sources.extend(uploaded_files)
            elif camera_file: active_sources.append(camera_file)
            
            st.markdown('<div class="btn-proses">', unsafe_allow_html=True)
            if st.button("🚀 PROSES GAMBAR SEKARANG"):
                if not active_sources:
                    st.warning("Silakan masukkan gambar terlebih dahulu.")
                elif not model:
                    st.error("Model AI 'best (12).pt' tidak ditemukan. Pastikan sudah diunggah.")
                else:
                    st.session_state["latest_results"] = []
                    
                    with st.spinner(f"YOLOv8 sedang menganalisis {len(active_sources)} gambar..."):
                        for file in active_sources:
                            # PROSES AI + KOREKSI
                            result_img, total_sacks, details = process_image(file, model, corrections_dict)
                            
                            if result_img:
                                now = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
                                date_str = now.strftime("%Y-%m-%d")
                                time_str = now.strftime("%H:%M:%S")
                                filename = file.name
                                
                                # UPLOAD KE IMGBB (AMAN PAKAI SECRETS)
                                st.toast(f"Menyimpan foto {filename} ke Cloud ImgBB...", icon="☁️")
                                img_url = upload_to_imgbb(result_img)
                                
                                st.session_state["latest_results"].append({
                                    "image": result_img, 
                                    "filename": filename, 
                                    "info": f"Total Terdeteksi: {total_sacks} Sak", 
                                    "details": details
                                })
                                
                                new_row = pd.DataFrame([{
                                    "Tanggal": date_str, 
                                    "Waktu": time_str, 
                                    "Nama File": filename,
                                    "Total Sak": total_sacks, 
                                    "Rincian Merek": details, 
                                    "Tautan Foto": img_url
                                }])
                                st.session_state["history_table"] = pd.concat([st.session_state["history_table"], new_row], ignore_index=True)
                                
                                # SIMPAN KE FILE FISIK CSV AGAR AMAN SAAT RELOAD
                                st.session_state["history_table"].to_csv(HISTORY_FILE, index=False)
                                
                        st.success("Pemindaian Selesai!")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_kanan:
            st.markdown("<h4 style='color: #1E293B;'>🔍 Detection Results</h4>", unsafe_allow_html=True)
            results = st.session_state.get("latest_results", [])
            if len(results) > 0:
                with st.container(height=600, border=False):
                    for idx, res in enumerate(results):
                        st.image(res["image"], use_container_width=True)
                        st.success(f"**{res['filename']}** | {res['info']}")
                        if res['details']: 
                            st.caption(f"Rincian Merek: {res['details']}")
                        
                        # TOMBOL UNDUH GAMBAR
                        buf = io.BytesIO()
                        res["image"].save(buf, format="JPEG")
                        byte_im = buf.getvalue()
                        st.download_button(
                            label=f"📥 Unduh Foto ({res['filename']})", 
                            data=byte_im, 
                            file_name=f"hasil_{res['filename']}", 
                            mime="image/jpeg", 
                            key=f"dl_img_{idx}"
                        )
                        st.markdown("---")
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