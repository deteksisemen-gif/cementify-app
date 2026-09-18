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
st.set_page_config(page_title="Cementify | Smart Inventory", layout="wide", initial_sidebar_state="collapsed")

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
# 4. HALAMAN LOGIN (DESAIN SPLIT SCREEN)
# ==========================================
def show_login_page():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .block-container { padding: 3rem 2rem !important; max-width: 100% !important; overflow: hidden; }
        #MainMenu, header, footer {visibility: hidden;}
        .left-panel { padding: 2% 8%; }
        .stTextInput>div>div>input { border-radius: 8px !important; padding: 10px 15px !important; }
        </style>
    """, unsafe_allow_html=True)

    col_kiri, col_kanan = st.columns([1, 1.2], gap="large")

    with col_kiri:
        st.markdown("<div class='left-panel'>", unsafe_allow_html=True)
        st.markdown("""
            <h3 style='margin:0; font-weight: 800; color: #0F172A; font-size: 24px; margin-bottom: 25px;'>Cementify ⚙️</h3>
            <h1 style='font-size: 38px; color: #0F172A; font-weight: 800; margin-bottom: 5px;'>Welcome Back!</h1>
            <p style='color: #64748B; margin-bottom: 25px;'>Please login to manage your inventory</p>
        """, unsafe_allow_html=True)

        username = st.text_input("Username / Email", placeholder="admin@cementify.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        st.markdown("<div style='text-align: right; color: #64748B; font-size: 14px; margin-top: -10px; margin-bottom: 20px; cursor: pointer;'>Forgot Password?</div>", unsafe_allow_html=True)

        # Tombol Login Lebar Penuh
        if st.button("LOGIN", type="primary", use_container_width=True):
            if password == "indocement123":
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("Kredensial tidak valid.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_kanan:
        # Trik Base64 untuk mengubah foto lokal menjadi Wallpaper proporsional
        img_path = "gambar ilustrasi web.jpg"
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                img_data = f.read()
            img_b64 = base64.b64encode(img_data).decode()
            
            st.markdown(f"""
                <div style="
                    background-image: url('data:image/jpeg;base64,{img_b64}'); 
                    background-size: cover; 
                    background-position: center; 
                    height: 70vh; 
                    width: 100%;
                    border-radius: 20px; 
                    box-shadow: -10px 0 20px rgba(0,0,0,0.1);
                ">&nbsp;</div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Foto 'gambar ilustrasi web.jpg' tidak ditemukan di GitHub.")

# ==========================================
# 5. HALAMAN DASHBOARD UTAMA
# ==========================================
def show_dashboard_page():
    st.markdown("""
        <style>
        [data-testid='stSidebar'] {display: block;}
        .block-container { padding: 3rem 5rem !important; }
        .btn-proses>button {
            background-color: #F5A623 !important; color: #1E232F !important; font-weight: 700 !important;
            border-radius: 8px !important; width: 100% !important; border: none !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## ⚡ Cementify")
        st.markdown("---")
        menu = st.radio("Navigasi Utama", ["Area Pemindaian", "Riwayat Data (CSV)"])
        st.markdown("---")
        if st.button("Keluar (Log Out)", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()

    model = load_vision_model()

    if menu == "Area Pemindaian":
        st.title("Area Pemindaian Inventori")
        st.markdown("Unggah foto gudang atau gunakan kamera perangkat untuk memulai deteksi.")
        st.markdown("---")
        
        col_input, col_result = st.columns([1, 1.2], gap="large")
        
        with col_input:
            st.markdown("#### 1. Input Visual")
            input_method = st.radio("Pilih Sumber:", ("Unggah Berkas (JPG/PNG)", "Gunakan Kamera"), horizontal=True)
            
            if input_method == "Unggah Berkas (JPG/PNG)":
                active_source = st.file_uploader("Format: JPG, JPEG, PNG, WEBP", type=['jpg', 'jpeg', 'png', 'webp'], label_visibility="collapsed")
            else:
                active_source = st.camera_input("Ambil Foto Lapangan", label_visibility="collapsed")
            
            st.markdown('<div class="btn-proses">', unsafe_allow_html=True)
            if st.button("Proses Gambar Sekarang"):
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

        with col_result:
            st.markdown("#### 2. Hasil Deteksi Visual")
            if st.session_state.get('latest_scan') is not None:
                st.info(f"**{st.session_state['latest_info']}**")
                st.image(st.session_state['latest_scan'], use_container_width=True, caption="Visualisasi Segmentasi Instance")
            else:
                st.info("Menunggu masukan gambar dari operator.")

    elif menu == "Riwayat Data (CSV)":
        st.title("Log Riwayat Sesi Aktif")
        st.markdown("Semua data pemindaian selama aplikasi berjalan akan direkam di sini.")
        st.markdown("---")
        
        st.dataframe(st.session_state["history_table"], hide_index=True, use_container_width=True)
        
        if not st.session_state["history_table"].empty:
            st.markdown("<br>", unsafe_allow_html=True)
            csv = st.session_state["history_table"].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Unduh Data ke Excel (CSV)",
                data=csv,
                file_name=f'Database_Semen_{datetime.datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
                use_container_width=True
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