import streamlit as st
from groq import Groq
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile
import os
import time

# --- CONFIGURACIÓN DE PÁGINA Y ESTILO ---
st.set_page_config(page_title="Traductor Despacho IA", layout="wide")

st.markdown("""
    <style>
    .chat-container { display: flex; flex-direction: column; gap: 10px; padding: 15px; background-color: #f0f2f5; border-radius: 10px; }
    
    /* Burbuja Profesional (Tú) - Izquierda */
    .bubble-me {
        background-color: #ffffff; padding: 12px 18px; border-radius: 18px 18px 18px 0px;
        max-width: 75%; align-self: flex-start; border: 1px solid #e0e0e0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1); margin-bottom: 5px;
    }
    
    /* Burbuja Cliente - Derecha */
    .bubble-client {
        background-color: #dcf8c6; padding: 12px 18px; border-radius: 18px 18px 0px 18px;
        max-width: 75%; align-self: flex-end; text-align: right;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1); margin-bottom: 5px;
    }
    
    .rtl { direction: rtl; }
    .translation-text { font-style: italic; color: #444; border-top: 1px solid rgba(0,0,0,0.05); margin-top: 8px; padding-top: 5px; font-size: 0.95em; }
    .tag { font-size: 0.7em; font-weight: bold; color: #888; margin-bottom: 4px; display: block; }
    </style>
    """, unsafe_allow_html=True)

# --- INICIALIZACIÓN ---
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except:
    st.error("⚠️ Configura la GROQ_API_KEY en 'Advanced Settings > Secrets'.")
    st.stop()

# Estados de la sesión
if "messages" not in st.session_state: st.session_state.messages = []
if "listening" not in st.session_state: st.session_state.listening = False
if "idioma_cliente" not in st.session_state: st.session_state.idioma_cliente = None

# --- LÓGICA DE TRADUCCIÓN ---
def get_translation(text, detected_lang, manual_lang):
    # Si hablas tú (español)
    if detected_lang == 'es':
        # Usa el idioma detectado del cliente si ya existe, si no el manual
        target = st.session_state.idioma_cliente if st.session_state.idioma_cliente else manual_lang
        prompt = f"Translate from Spanish to {target}. Return ONLY the translation: {text}"
        role = "me"
        display_lang = target
    else:
        # Habla el cliente (Cualquier otro idioma)
        st.session_state.idioma_cliente = detected_lang # El sistema "aprende" el idioma del cliente
        prompt = f"Traduce al español de forma profesional: {text}. Devuelve SOLO la traducción."
        role = "client"
        display_lang = "Español"
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return role, completion.choices[0].message.content, detected_lang

# --- INTERFAZ ---
st.title("🎙️ Intérprete Manos Libres de Despacho")

col_chat, col_control = st.columns([3, 1])

with col_control:
    st.subheader("⚙️ Configuración")
    
    # Idioma inicial (antes de que el cliente hable)
    idioma_inicial = st.selectbox("Idioma inicial del cliente", ["Inglés", "Árabe", "Francés", "Alemán", "Ruso", "Chino"])
    
    st.write("---")
    
    # Botón principal ON/OFF
    if not st.session_state.listening:
        if st.button("🔴 INICIAR ESCUCHA", use_container_width=True):
            st.session_state.listening = True
            st.rerun()
    else:
        if st.button("🛑 DETENER", use_container_width=True):
            st.session_state.listening = False
            st.rerun()

    if st.button("🗑️ Limpiar Sesión (Anonimizar)"):
        st.session_state.messages = []
        st.session_state.idioma_cliente = None
        st.rerun()

    if st.session_state.listening:
        st.success("Micro abierto. Hablad con normalidad...")
    else:
        st.info("Micro pausado.")

# --- BUCLE DE ESCUCHA ACTIVA ---
if st.session_state.listening:
    fs = 16000  # Frecuencia Whisper
    duration = 6 # Segundos por bloque (puedes ajustarlo a 8 si habláis lento)
    
    with st.spinner("Escuchando..."):
        # Grabación directa
        rec = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            write(tmp.name, fs, rec)
            
            try:
                with open(tmp.name, "rb") as f:
                    # Transcripción (STT)
                    ts = client.audio.transcriptions.create(
                        file=(tmp.name, f.read()),
                        model="whisper-large-v3",
                        response_format="verbose_json"
                    )
                
                # Solo procesamos si hay audio real
                if ts.text.strip() and len(ts.text) > 2:
                    role, translated, lang_code = get_translation(ts.text, ts.language, idioma_inicial)
                    st.session_state.messages.append({
                        "role": role,
                        "original": ts.text,
                        "translated": translated,
                        "lang": lang_code
                    })
                    os.remove(tmp.name)
                    st.rerun() # Reinicia el bucle para mostrar y seguir oyendo
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if os.path.exists(tmp.name): os.remove(tmp.name)

# --- RENDERIZADO DEL CHAT ---
with col_chat:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for m in st.session_state.messages:
        if m["role"] == "me":
            # Burbuja del profesional (Tú)
            target = st.session_state.idioma_cliente if st.session_state.idioma_cliente else idioma_inicial
            st.markdown(f'''
                <div class="bubble-me">
                    <span class="tag">TÚ (Español)</span>
                    {m["original"]}
                    <div class="translation-text"><b>{target}:</b> {m["translated"]}</div>
                </div>
            ''', unsafe_allow_html=True)
        else:
            # Burbuja del cliente
            rtl = "rtl" if m["lang"] == "ar" else ""
            st.markdown(f'''
                <div class="bubble-client">
                    <span class="tag">CLIENTE ({m["lang"].upper()})</span>
                    <span class="{rtl}">{m["original"]}</span>
                    <div class="translation-text"><b>Español:</b> {m["translated"]}</div>
                </div>
            ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
