import streamlit as st
from groq import Groq
import tempfile
import os
from audio_recorder_streamlit import audio_recorder

# 1. Configuración de la página y Estilo CSS
st.set_page_config(page_title="Traductor Pro Despacho", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0f2f5; }
    .chat-container { 
        display: flex; 
        flex-direction: column; 
        gap: 15px; 
        padding: 20px; 
        max-height: 70vh;
        overflow-y: auto;
    }
    .bubble-me {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 15px 15px 15px 0px;
        max-width: 80%;
        align-self: flex-start;
        box-shadow: 0px 1px 2px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        color: #303030;
    }
    .bubble-client {
        background-color: #dcf8c6;
        padding: 15px;
        border-radius: 15px 15px 0px 15px;
        max-width: 80%;
        align-self: flex-end;
        box-shadow: 0px 1px 2px rgba(0,0,0,0.1);
        color: #303030;
    }
    .rtl { direction: rtl; text-align: right; display: block; }
    .translation-sub { 
        font-size: 0.9em; 
        color: #555; 
        margin-top: 8px; 
        border-top: 1px solid rgba(0,0,0,0.05); 
        padding-top: 5px; 
        font-style: italic;
    }
    .lang-tag {
        font-size: 0.7em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 4px;
        display: block;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Inicialización de API
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error("Error: Configura GROQ_API_KEY en Advanced Settings > Secrets.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🌐 Traductor de Despacho en Tiempo Real")

# 3. Función de procesamiento
def process_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    try:
        with open(temp_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_path, file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
        
        detected_lang = transcription.language
        text = transcription.text

        if detected_lang == "es":
            st.session_state.messages.append({"role": "me", "text": text})
        else:
            # CAMBIO AQUÍ: Usamos el modelo llama-3.3-70b-versatile que es el actual
            translation_prompt = f"Traduce al español de forma profesional: '{text}'. Devuelve SOLO el texto traducido."
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": translation_prompt}]
            )
            translated_text = completion.choices[0].message.content
            st.session_state.messages.append({
                "role": "client", 
                "original": text, 
                "translated": translated_text, 
                "lang": detected_lang
            })

    except Exception as e:
        st.error(f"Error al procesar el audio: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# 4. Interfaz
col_chat, col_control = st.columns([3, 1])

with col_control:
    st.write("### 🎙️ Controles")
    audio_data = audio_recorder(
        text="Pulsa para hablar",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_size="3x",
    )
    
    if audio_data:
        with st.spinner("Traduciendo..."):
            process_audio(audio_data)

    if st.button("🗑️ Limpiar Sesión"):
        st.session_state.messages = []
        st.rerun()

with col_chat:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "me":
            st.markdown(f'<div class="bubble-me"><span class="lang-tag">Tú</span>{msg["text"]}</div>', unsafe_allow_html=True)
        else:
            is_rtl = "rtl" if msg["lang"] == "ar" else ""
            st.markdown(f'''
                <div class="bubble-client">
                    <span class="lang-tag">Cliente ({msg["lang"].upper()})</span>
                    <span class="{is_rtl}">{msg["original"]}</span>
                    <div class="translation-sub"><b>Traducción:</b> {msg["translated"]}</div>
                </div>
            ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
