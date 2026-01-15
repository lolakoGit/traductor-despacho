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
    
    /* Contenedor de burbujas */
    .chat-container { 
        display: flex; 
        flex-direction: column; 
        gap: 15px; 
        padding: 20px; 
        max-height: 70vh;
        overflow-y: auto;
    }
    
    /* Burbuja del Profesional (Tú) - Izquierda (Blanco/Gris claro) */
    .bubble-me {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 15px 15px 15px 0px;
        max-width: 80%;
        align-self: flex-start;
        box-shadow: 0px 1px 2px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        color: #303030;
        font-family: sans-serif;
    }
    
    /* Burbuja del Cliente - Derecha (Verde claro) */
    .bubble-client {
        background-color: #dcf8c6;
        padding: 15px;
        border-radius: 15px 15px 0px 15px;
        max-width: 80%;
        align-self: flex-end;
        box-shadow: 0px 1px 2px rgba(0,0,0,0.1);
        color: #303030;
        font-family: sans-serif;
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

# 2. Inicialización de API y Estado de Sesión
# Asegúrate de tener GROQ_API_KEY en los Secrets de Streamlit
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error("Error: No se encontró la API Key en los Secrets.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Interfaz de Usuario
st.title("🌐 Traductor de Despacho en Tiempo Real")

# 3. Función para procesar audio y traducir
def process_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    try:
        # Transcripción con Whisper Large V3 (detecta idioma automáticamente)
        with open(temp_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_path, file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
        
        detected_lang = transcription.language
        text = transcription.text

        # Lógica de clasificación:
        # Si detecta español (es), se asume que eres tú hablando.
        if detected_lang == "es":
            st.session_state.messages.append({"role": "me", "text": text})
        else:
            # Si no es español, traducimos usando Llama 3
            translation_prompt = f"Translate the following text to Spanish. Output ONLY the translation: '{text}'"
            completion = client.chat.completions.create(
                model="llama3-8b-8192",
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

# 4. Layout de la aplicación
col_chat, col_control = st.columns([3, 1])

with col_control:
    st.write("### 🎙️ Controles")
    # Componente de grabación
    audio_data = audio_recorder(
        text="Pulsa para hablar",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_size="3x",
    )
    
    if audio_data:
        with st.spinner("Procesando..."):
            process_audio(audio_data)

    st.write("---")
    if st.button("🗑️ Limpiar y Anonimizar"):
        st.session_state.messages = []
        st.rerun()

with col_chat:
    # Contenedor visual de la conversación
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for msg in st.session_state.messages:
        if msg["role"] == "me":
            # Burbuja tuya (Español)
            st.markdown(f'''
                <div class="bubble-me">
                    <span class="lang-tag">Tú (Español)</span>
                    {msg["text"]}
                </div>
            ''', unsafe_allow_html=True)
        else:
            # Burbuja del cliente (Traducida)
            # Si es árabe (ar), aplicamos la clase RTL
            is_rtl = "rtl" if msg["lang"] == "ar" else ""
            st.markdown(f'''
                <div class="bubble-client">
                    <span class="lang-tag">Cliente ({msg["lang"].upper()})</span>
                    <span class="{is_rtl}">{msg["original"]}</span>
                    <div class="translation-sub"><b>Traducción:</b> {msg["translated"]}</div>
                </div>
            ''', unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)
