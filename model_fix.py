import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm
import pickle
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

st.set_page_config(page_title="Emotion Recognition", layout="wide")

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rakkas&family=Poppins:wght@400;600&display=swap');
    #MainMenu, footer, header { visibility: hidden; }

    /* Animasi untuk gradien background */
    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Animasi untuk konten  */
    @keyframes fadeInRise {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Warna untuk background  */
    .stApp {
        background: linear-gradient(135deg, #e0c3fc, #8ec5fc, #fbc2eb, #a6c1ee); 
        background-size: 400% 400%; 
        animation: gradient-animation 15s ease infinite; /* Menerapkan animasi */
    }
    
    /* Judul utama  */
    .main-title {
        font-family: 'Rakkas', serif !important;
        font-size: 80px;
        text-align: center;
        margin: 100px 0 160px 0;
        color: #4a0e6b; 
    }
    
    /* Judul halaman (About, Help, Detect) */
    .page-title {
        font-family: 'Rakkas', serif !important;
        font-size: 60px;
        color: #4a0e6b !important; 
        text-align: center;
        margin: 20px 0 10px 0;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.1); 
    }
    
    /* Sub Judul */
    .subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: 36px;
        color: #333333; 
        text-align: center;
        margin: 20px 0;
    }
    
    /* Button style  */
    .stButton > button {
        background-color: #9279ff; 
        color: white;
        font-family: 'Poppins', sans-serif;
        font-size: 20px;
        font-weight: 600;
        padding: 12px 35px;
        border: none;
        border-radius: 30px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
        transition: all 0.3s ease;
        white-space: nowrap !important; 
        transform: scale(1); 
    }
    
    .stButton > button:hover {
        background-color: #7a5fff; 
        color: white;
        border: none;
        transform: scale(1.05); 
        box-shadow: 0 6px 20px rgba(0,0,0,0.2); 
    }
    
    /* Button style for close */
    .stButton > button[key*="close"] {
        background-color: #ff8a8a; 
        padding: 5px 15px;
        font-size: 16px;
        border-radius: 20px; 
    }
    .stButton > button[key*="close"]:hover {
        background-color: #ff6b6b;
        transform: scale(1.05);
    }
    
    /* Teks  */
    .content-text {
        font-family: 'Poppins', sans-serif;
        font-size: 18px;
        color: #333333 !important; 
        text-align: center;
        line-height: 1.8;
        max-width: 900px;
        margin: 30px auto;
        padding: 0 20px;
    }
    
    .custom-hr {
        border: none;
        height: 2px;
        background: linear-gradient(to right, transparent, #4a0e6b, transparent); 
        margin: 20px auto;
        max-width: 1100px;
    }
    
    /* Style untuk upload file  */
    .stFileUploader {
        max-width: 400px;
        margin: 100px auto 0 auto;
    }
    .stFileUploader > div > button {
        background-color: rgba(255, 255, 255, 0.7) !important; 
        color: #4a0e6b !important; 
        font-family: 'Poppins', sans-serif !important;
        font-size: 28px !important;
        font-weight: 400 !important;
        padding: 40px 80px !important;
        border: 3px dashed #9279ff !important; 
        border-radius: 20px !important; 
        width: 100%;
        transition: all 0.3s ease; 
    }
    .stFileUploader > div > button:hover {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border: 3px dashed #7a5fff !important; 
        color: #4a0e6b !important;
    }

    .content-block {
        animation: fadeInRise 0.8s ease-out; 
        background-color: rgba(255, 255, 255, 0.4);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin-top: 20px;
    }

    /* Layar Responsif  */
    @media (max-width: 768px) {
        .main-title {
            font-size: 50px;
            margin: 50px 0 80px 0;
        }
        .page-title {
            font-size: 40px;
        }
        .subtitle {
            font-size: 28px;
        }
        .content-text {
            font-size: 16px;
        }
        .stButton > button {
            font-size: 16px;
            padding: 10px 25px;
        }
        .stFileUploader > div > button {
            font-size: 22px !important;
            padding: 30px 60px !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# Fungsi untuk memuat model
@st.cache_resource
def load_model_assets():
    try:
        with open('model_best_loss.pkl', 'rb') as f:
            data = pickle.load(f)

        model_state_dict = data['model_state_dict']
        class_to_int = data['class_to_int']
        model_architecture = data['model_architecture']
        
        num_classes = len(class_to_int)
        model = timm.create_model(model_architecture, pretrained=False, num_classes=num_classes)
        
        model.load_state_dict(model_state_dict)
        model.eval()
        
        int_to_class = {v: k for k, v in class_to_int.items()}
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return model, int_to_class, face_cascade
    
    except FileNotFoundError:
        st.error("File 'model_best_loss.pkl' tidak ditemukan.")
        return None, None, None
    except Exception as e:
        st.error(f"Terjadi error saat memuat model: {e}")
        return None, None, None

model, int_to_class, face_cascade = load_model_assets()

# Fungsi untuk melakukan prediksi
def predict_and_draw(image_np, model, int_to_class, face_cascade):
    image_to_draw = image_np.copy()
    gray_image = cv2.cvtColor(image_to_draw, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    color_map = {
        'happy': (3, 138, 16),   
        'sad': (255, 0, 0),       
        'neutral': (128, 0, 128), 
        'angry': (0, 0, 255)      
    }
    
    default_color = (100, 100, 100) 
    
    if len(faces) == 0:
        st.warning("Tidak ada wajah yang terdeteksi pada gambar.")
    

    for (x, y, w, h) in faces:
        face_roi = image_to_draw[y:y+h, x:x+w]
        
        # Preprocessing wajah 
        resized_face = cv2.resize(face_roi, (224, 224))
        rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
        tensor_face = torch.from_numpy(rgb_face.astype(np.float32) / 255.0).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor_face = (tensor_face - mean) / std
        tensor_face = tensor_face.unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(tensor_face)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        pred_class = int_to_class[predicted_idx.item()]
        confidence_score = confidence.item()
        
        box_color = color_map.get(pred_class, default_color)
        cv2.rectangle(image_to_draw, (x, y), (x+w, y+h), box_color, 2)
        
        label = f"{pred_class}: {confidence_score:.1%}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        cv2.rectangle(image_to_draw, (x, y - text_height - 10), (x + text_width, y), box_color, -1)
        cv2.putText(image_to_draw, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    return image_to_draw

# Class untuk memproses video real-time
class EmotionVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.int_to_class = int_to_class
        self.face_cascade = face_cascade

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        processed_img = predict_and_draw(img, self.model, self.int_to_class, self.face_cascade)
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")



if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'detect_mode' not in st.session_state:
    st.session_state.detect_mode = 'upload' 

# Halaman Home
if st.session_state.page == 'home':
    col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
    with col1:
        if st.button("Help", key="help_btn", use_container_width=True):
            st.session_state.page = 'help'
            st.rerun()
    with col3:
        if st.button("About", key="about_btn", use_container_width=True):
            st.session_state.page = 'about'
            st.rerun()
            
    st.markdown("<div style='height: 90px;'></div>", unsafe_allow_html=True)

    st.markdown('<h1 class="main-title">Emotion Recognition</h1>', unsafe_allow_html=True)
    
    st.markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)
    
    col1_detect, col2_detect, col3_detect = st.columns([2, 1, 2])
    with col2_detect:
        if st.button("Detect", key="detect_btn", use_container_width=True):
            st.session_state.page = 'detect'
            st.rerun()

# Halaman About
elif st.session_state.page == 'about':
    col1, col2 = st.columns([0.9, 0.1])
    with col2:
        if st.button("X", key="close_about"):
            st.session_state.page = 'home'
            st.rerun()
    
    st.markdown('<div class="content-block">', unsafe_allow_html=True) 
    
    st.markdown('<h1 class="page-title">Emotion Recognition</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="subtitle">About</h2>', unsafe_allow_html=True)
    st.markdown('<hr class="custom-hr">', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="content-text" style="text-align: center; max-width: 800px;">
    <p class="content-text">
    Emotion Recognition merupakan sebuah sistem sederhana yang berfungsi untuk mendeteksi emosi manusia berdasarkan ekspresi wajahnya. Sistem ini dapat melakukan klasifikasi pada 4 emosi yaitu senang (happy), sedih (sad), marah (angry), dan biasa saja (neutral).
    </p>
    ''', unsafe_allow_html=True)
    
    st.markdown('<h2 class="subtitle">About Me</h2>', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="content-text" style="text-align: left; max-width: 800px;">
    <p class="content-text">
    Juventia Agnecia merupakan mahasiswa semester 7 yang saat ini sedang menjalankan skripsi pada bidang Intelligent System dengan dosen pembimbingnya yaitu Ibu Dra. Charisni Lubis, M.Kom dengan judul topik "Prediksi Emosi Manusia Menggunakan Swin Transformer Berdasarkan Ekspresi Wajah"
    </p>
    ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True) 

# Halaman Help
elif st.session_state.page == 'help':
    col1, col2 = st.columns([0.9, 0.1])
    with col2:
        if st.button("X", key="close_help"):
            st.session_state.page = 'home'
            st.rerun()
    
    st.markdown('<div class="content-block">', unsafe_allow_html=True)
    
    st.markdown('<h1 class="page-title">Emotion Recognition</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="subtitle">Help</h2>', unsafe_allow_html=True)
    st.markdown('<hr class="custom-hr">', unsafe_allow_html=True)
    
    st.markdown('<h3 style="font-family: \'Poppins\', sans-serif; font-size: 32px; text-align: center; margin-top: 50px;">How To Run the Program</h3>', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="content-text" style="text-align: left; max-width: 800px;">
    <ol style="font-size: 18px; line-height: 2;">
        <li>Klik tombol "Detect" untuk memasuki page deteksi emosi</li>
        <li>Klik tombol "Upload Image" untuk melakukan deteksi pada foto </li>
        <li>Klik "Browse Files" untuk mencari foto wajah yang ingin dideteksi</li>
        <li>Untuk mendeteksi secara Real Time, klik tombol "Real Time" di samping kanan tombol "Upload Image"</li>
        <li>Tekan tombol "Start" untuk mendeteksi emosi secara Real Time</li>
    </ol>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True) 

# Halaman Detect
elif st.session_state.page == 'detect':
    col1, col2 = st.columns([0.9, 0.1])
    with col2:
        if st.button("X", key="close_detect"):
            st.session_state.page = 'home'
            st.rerun()

    st.markdown('<div class="content-block">', unsafe_allow_html=True)
    
    st.markdown('<h1 class="page-title">Emotion Recognition</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="subtitle">Detect</h2>', unsafe_allow_html=True)
    
    col_spacer1, col_btn1, col_btn2, col_spacer2 = st.columns([2.5, 1.5, 1.5, 2.5])     
    with col_btn1:
        if st.button("Upload Image", use_container_width=True):
            st.session_state.detect_mode = 'upload'
            st.rerun()
    with col_btn2:
        if st.button("Real time", use_container_width=True):
            st.session_state.detect_mode = 'real_time'
            st.rerun()
            
    st.markdown('<hr class="custom-hr">', unsafe_allow_html=True)

    if model is not None:
        if st.session_state.detect_mode == 'upload':
           
            # Tampilkan UI untuk upload gambar
            uploaded_file = st.file_uploader(
                "Upload Image", 
                type=['png', 'jpg', 'jpeg'], 
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                image_np = np.array(image)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                result_image = predict_and_draw(image_bgr, model, int_to_class, face_cascade)
                result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

                col_spacer1, col_img, col_spacer2 = st.columns([1, 2, 1])
                with col_img:
                    st.image(result_image_rgb, use_container_width=True)

        
        elif st.session_state.detect_mode == 'real_time':
            st.info("Klik 'START' untuk menyalakan kamera dan deteksi emosi secara real-time.")
            
            webrtc_streamer(
                key="emotion-detection-webrtc",
                video_processor_factory=EmotionVideoTransformer,
                rtc_configuration=RTCConfiguration({
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                }),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
    else:
        st.error("Model tidak berhasil dimuat, fungsi deteksi tidak tersedia.")
        
    st.markdown('</div>', unsafe_allow_html=True) 
    
    
    