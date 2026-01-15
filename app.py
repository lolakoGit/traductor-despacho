import streamlit as st
from groq import Groq
import tempfile
import os
from audio_recorder_streamlit import audio_recorder

# 1. Configuración de la página y Estilo CSS (WhatsApp Style)
st.set_page_config(page_title="Traductor Pro Despacho", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0f2f5; }
    .stButton>button { width: 100%; border-radius: 20px; }
    
    /* Contenedor de burbujas */
    .chat-container { display: flex; flex-direction: column; gap: 10px; padding: 20px; }
    
    /* Burbuja del Profesional (Tú) - Izquierda */
    .bubble-me {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 15px 15px 15px 0px;
        max-width: 70%;
        align-self: flex-start;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    /* Burbuja del Cliente - Derecha */
    .bubble-client {
        background-color: #dcf8c6;
        padding: 15px;
        border-radius: 15px 15px 0px 15px;
        max-width: 70%;
        align-self: flex-end;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
        text-align: right;
    }
    
    .rtl { direction: rtl; unicode-bidi: bidi-override; }
    .translation-sub { font-size: 0.85em; color: #555; margin-top: 5px; border-top: 1px dashed #ccc; padding-top: 5px; }
    </style>
    """, unsafe_layout=True)

# 2. Inicialización de API y Estado
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

# Título
st.title("🌐 Traductor Inteligente de Despacho")
st.info("El micrófono captará el audio, detectará el idioma y lo traducirá automáticamente.")

# 3. Función para procesar audio y traducir
def process_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    try:
        # Transcripción con Whisper Large V3
        with open(temp_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_path, file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
        
        detected_lang = transcription.language
        text = transcription.text

        # Si el idioma no es español, traducimos al español usando Llama 3
        if detected_lang != "es":
            translation_prompt = f"Traducción fiel al español de este texto en {detected_lang}: '{text}'. Solo devuelve el texto traducido."
            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": translation_prompt}]
            )
            translated_text = completion.choices[0].message.content
            st.session_state.messages.append({"role": "client", "original": text, "translated": translated_text, "lang": detected_lang})
        else:
            st.session_state.messages.append({"role": "me", "text": text})

    finally:
        os.remove(temp_path)

# 4. Interfaz de Usuario
col_chat, col_control = st.columns([3, 1])

with col_control:
    st.write("### 🎙️ Control de Voz")
    audio_data = audio_recorder(
        text="Pulsa para grabar",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_size="3x",
    )
    
    if audio_data:
        process_audio(audio_data)

    if st.button("🗑️ Limpiar Sesión (Anonimizar)"):
        st.session_state.messages = []
        st.rerun()

with col_chat:
    st.write("### 💬 Conversación")
    
    # Mostrar mensajes con formato de burbujas
    st.markdown('<div class="chat-container">', unsafe_layout=True)
    for msg in st.session_state.messages:
        if msg["role"] == "me":
            st.markdown(f'''
                <div class="bubble-me">
                    <b>Tú:</b><br>{msg["text"]}
                </div>
            ''', unsafe_layout=True)
        else:
            # Si es árabe, aplicamos clase RTL
            rtl_class = "rtl" if msg["lang"] == "ar" else ""
            st.markdown(f'''
                <div class="bubble-client">
                    <b>Cliente ({msg["lang"].upper()}):</b><br>
                    <span class="{rtl_class}">{msg["original"]}</span>
                    <div class="translation-sub"><b>Traducción:</b> {msg["translated"]}</div>
                </div>
            ''', unsafe_layout=True)
    st.markdown('</div>', unsafe_layout=True)
