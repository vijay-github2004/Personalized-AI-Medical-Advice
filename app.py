import streamlit as st
import easyocr
import google.generativeai as genai
import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from langdetect import detect
from googletrans import Translator
import speech_recognition as sr
import os
import subprocess  # For text-to-speech
from gtts import gTTS
import tempfile

# ✅ Configure Gemini API Keys
API_KEYS = [
    "AIzaSyAsXaGluwAGWAcO-RiiIwpRT-TQvU6AiCg",  # First API Key
    "AIzaSyA-YPH_WppJawH4D_hNIrDShD9rn1MTfRg"   # Second API Key
]

current_api_index = 0  # Start with the first API key

def switch_api_key():
    """Switch to the next API key when one is exhausted."""
    global current_api_index
    current_api_index = (current_api_index + 1) % len(API_KEYS)  # Rotate between keys
    genai.configure(api_key=API_KEYS[current_api_index])  # Update API key

# ✅ Initialize the first API key
genai.configure(api_key=API_KEYS[current_api_index])

# ✅ Load EasyOCR reader
reader = easyocr.Reader(['en'])
translator = Translator()
recognizer = sr.Recognizer()

# ✅ Check GPU Availability
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"✅ Device in use: {device}")

# ✅ Function: Detect Language and Translate
def translate_text(text, target_lang="en"):
    detected_lang = detect(text)
    if detected_lang != target_lang:
        return translator.translate(text, src=detected_lang, dest=target_lang).text
    return text

# ✅ Function: Extract Text from Image
def extract_text(image):
    result = reader.readtext(np.array(image), detail=0)
    extracted_text = " ".join(result) if result else "No text detected."
    return translate_text(extracted_text)

# ✅ Function: Identify Medicine Name using Gemini API
def identify_medicine(extracted_text):
    """Extracts medicine name from text using Gemini API with auto key switching."""
    for _ in range(len(API_KEYS)):  # Try both API keys
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            prompt = f"""
            From the following text, extract the name of the medicine or tablet:
            "{extracted_text}"
            Return only the medicine name.
            """
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if "429" in str(e):  # If quota exceeded, switch API key
                switch_api_key()
            else:
                return f"❌ Error identifying medicine: {e}"

    return "❌ All API keys exhausted. Try again later."

# ✅ Function: Get Medicine Details
def get_medicine_details(medicine_name, patient_info, lang="en"):
    """Fetches medicine details with auto API switching."""
    for _ in range(len(API_KEYS)):  # Try both API keys
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            prompt = f"""
            A patient needs information about the medicine "{medicine_name}". Here are their details:

            - Age: {patient_info['Age']} years
            - Weight: {patient_info['Weight']} kg
            - Gender: {patient_info['Gender']}
            - Previous Conditions: {patient_info['Previous Disease History']}
            - Current Illness: {patient_info['Current Disease']}
            - Other Tablets Consumed Regularly: {patient_info['Other Tablets']}
            
            Provide a detailed yet easy-to-understand response covering:
            - How {medicine_name} works and what it is used for
            - Recommended dosage and administration guidelines
            - Possible side effects
            - Interactions with other medications
            - Additional precautions
            - Alternative medicines
            - Suggested diet and clothing precautions

            Ensure the explanation is patient-friendly.
            """
            response = model.generate_content(prompt)
            translated_response = translator.translate(response.text, dest=lang).text
            return translated_response if translated_response else "No details available."
        except Exception as e:
            if "429" in str(e):  # If quota exceeded, switch API key
                switch_api_key()
            else:
                return f"❌ Error fetching details: {e}"

    return "❌ All API keys exhausted. Try again later."

# ✅ Function: Chatbot Response
def chatbot_response(user_input, medicine_name, patient_info, lang="en"):
    """Handles chatbot responses using Gemini API with auto key switching."""
    for _ in range(len(API_KEYS)):  # Try both API keys
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            prompt = f"""
            Patient's Question: "{user_input}"
            
            Medicine Details:
            - Name: {medicine_name}
            
            Patient Details:
            - Age: {patient_info.get('Age', 'Unknown')} years
            - Weight: {patient_info.get('Weight', 'Unknown')} kg
            - Gender: {patient_info.get('Gender', 'Unknown')}
            - Current Illness: {patient_info.get('Current Disease', 'Not specified')}
            - Other Medications: {patient_info.get('Other Tablets', 'None')}
            
            Instructions for AI:
            - Provide a concise, clear, and patient-friendly answer.
            - Ensure the response is relevant to the medicine and patient details.
            - Avoid giving general medical advice or instructions without context.
            """
            response = model.generate_content(prompt)
            return translate_text(response.text, lang)
        except Exception as e:
            if "429" in str(e):  # If quota exceeded, switch API key
                switch_api_key()
            else:
                return f"❌ Error: {e}"

    return "❌ All API keys exhausted. Try again later."

# ✅ Function: Voice Input

# Initialize the recognizer globally
recognizer = sr.Recognizer()

def voice_input():
    with sr.Microphone() as source:
        st.write("🎧 Speak now...")

        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)

        try:
            # Listen for user input with a timeout
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio."
        except sr.RequestError:
            return "Speech recognition service is unavailable."
        except Exception as e:
            return f"Error: {e}"



# ✅ Streamlit UI
st.title("🩺 Personalized AI Medical Advice")
st.write("Upload an image to scan medicine names and get personalized medical advice.")

# ✅ Step 1: Upload or Capture Image
st.markdown("### 📸 Upload or Capture an Image")
use_camera = st.checkbox("Use Webcam to Capture Image")

image = None
if use_camera:
    camera_image = st.camera_input("Capture an Image")
    if camera_image:
        image = Image.open(camera_image)
else:
    uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

# ✅ Step 2: Show Extracted Text & Medicine Name
medicine_name = "Unknown Medicine"
if image:
    st.image(image, caption="📷 Uploaded Image", use_column_width=True, output_format="auto")
    extracted_text = extract_text(image)
    st.subheader("📝 Extracted Text (Translated)")
    st.write(extracted_text)

    medicine_name = identify_medicine(extracted_text)
    st.subheader("🔍 Detected Medicine")
    st.success(medicine_name)

# ✅ Step 3: Show Patient Information Form if Medicine is Detected
patient_info = {}
if medicine_name != "Unknown Medicine":
    st.subheader("🧑‍⚕️ Patient Information")
    user_lang = st.text_input("🌍 Preferred Language (e.g., en, ta, hi)", "en")

    translated_labels = {
        "Age": translate_text("Age", user_lang),
        "Weight": translate_text("Weight (kg)", user_lang),
        "Gender": translate_text("Gender", user_lang),
        "Previous Disease History": translate_text("Any previous diseases?", user_lang),
        "Current Disease": translate_text("What are you currently suffering from?", user_lang),
        "Other Tablets": translate_text("Any other tablets consumed regularly?", user_lang)
    }

    use_voice = st.checkbox("🎤 Use Voice Input Instead of Typing")

    def get_input(label):
        return voice_input() if use_voice else st.text_input(f"🔹 {label}")

    patient_info = {
        "Age": get_input(translated_labels['Age']),
        "Weight": get_input(translated_labels['Weight']),
        "Gender": st.radio(f"🔹 {translated_labels['Gender']}", ["Male", "Female", "Other"], horizontal=True),
        "Previous Disease History": get_input(translated_labels['Previous Disease History']),
        "Current Disease": get_input(translated_labels['Current Disease']),
        "Other Tablets": get_input(translated_labels['Other Tablets'])
    }

# ✅ Step 4: Get Medicine Details
if medicine_name != "Unknown Medicine" and patient_info and st.button("🔎 Get Medicine Details", key="get_medicine_details"):
    details = get_medicine_details(medicine_name, patient_info, user_lang)
    
    # ✅ Store details in session state
    st.session_state["medicine_details"] = details

# ✅ Step 5: Always Show Medicine Details If Available
if "medicine_details" in st.session_state and st.session_state["medicine_details"]:
    st.subheader("📋 Medicine Details & Suitability")
    st.info(st.session_state["medicine_details"])

    # ✅ Show "Read Aloud" Button After Details are Fetched
    if st.button("🔊 Read Aloud", key="read_aloud"):
        with st.spinner("🔄 Generating audio... Please wait"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                    tts = gTTS(st.session_state["medicine_details"], lang=user_lang)
                    temp_audio_path = temp_audio.name
                    tts.save(temp_audio_path)
                
                st.audio(temp_audio_path, format="audio/mp3")
                st.success("🔊 Reading aloud...")
            except Exception as e:
                st.error(f"❌ Error: {e}")


# ✅ Chatbot Section (Always Visible)
st.subheader("🤖 Chat with AI Doctor")
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

user_query = st.text_input(
    "💬 Any doubts? Chat with me:",
    placeholder="Type your question here...",
    key="chat_input"
)

if st.button("Ask Now", key="ask_now"):
    if not user_query.strip():
        st.warning("⚠ Please enter a question.")
    else:
        chatbot_reply = chatbot_response(user_query, medicine_name, patient_info, user_lang)
        if chatbot_reply:
            st.markdown(f'<div class="chatbot-reply">🤖 AI Doctor: {chatbot_reply}</div>', unsafe_allow_html=True)
        else:
            st.error("❌ No response from AI. Please try again.")

st.markdown('</div>', unsafe_allow_html=True)

# ✅ Project Credits
st.markdown("---")
st.markdown("### **👨‍💻 Project Developed By:**")
st.markdown(" **VIJAY, P.K. VIGNESH, and MOHAMMED FAIZAL**")
