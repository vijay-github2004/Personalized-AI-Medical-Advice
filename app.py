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

# ✅ Configure Gemini API
GEMINI_API_KEY = "AIzaSyB6wxK4bXYVc2P-6Xmrw8iJxwryMF7pOVc"  # Replace with actual API key
genai.configure(api_key=GEMINI_API_KEY)

# ✅ Load EasyOCR reader
reader = easyocr.Reader(['en'])  # Recognizes text only in English
translator = Translator()
recognizer = sr.Recognizer()
# ✅ Medicine Keywords for Detection
MEDICINE_KEYWORDS = [
    "Paracetamol", "Ibuprofen", "Aspirin", "Metformin", "Amoxicillin",
    "Ciprofloxacin", "Azithromycin", "Cetirizine", "Dolo", "Combiflam",
    "Happi", "Ursox 300 Tablet", "Ursox", "Ursomax", "Pyloflush",
    "Panzynorm HS", "Colospa X", "Colospa", "Prohance Liv", "Prohance",
    "Pantoprazole", "Rabeprazole", "Ranitidine"
]

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

# ✅ Function: Identify Medicine Name
def identify_medicine(extracted_text):
    for keyword in MEDICINE_KEYWORDS:
        if keyword.lower() in extracted_text.lower():
            return keyword
    return "Unknown Medicine"

def get_medicine_details(medicine_name, patient_info, lang="en"):
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
        - sugges what type of food eat and clothes

        Ensure the explanation is patient-friendly.
        """
        response = model.generate_content(prompt)
        translated_response = translator.translate(response.text, dest=lang).text
        return translated_response if translated_response else "No details available."
    except Exception as e:
        return f"❌ Error fetching details: {e}"
    
    # ✅ Function: Chatbot Response
def chatbot_response(user_input, medicine_name, patient_info, lang="en"):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        prompt = f"""
        Patient's Question: "{user_input}"
        
        **Medicine Details:**
        - Name: {medicine_name}
        
        **Patient Details:**
        - Age: {patient_info.get('Age', 'Unknown')} years
        - Weight: {patient_info.get('Weight', 'Unknown')} kg
        - Gender: {patient_info.get('Gender', 'Unknown')}
        - Current Illness: {patient_info.get('Current Disease', 'Not specified')}
        - Other Medications: {patient_info.get('Other Tablets', 'None')}
        
        **Instructions for AI:**
        - Provide a concise, clear, and patient-friendly answer.
        - Ensure the response is relevant to the medicine and patient details.
        - Avoid giving general medical advice or instructions without context.
        """
        response = model.generate_content(prompt)
        return translate_text(response.text, lang)
    except Exception as e:
        return f"❌ Error: {e}"

# ✅ Function: Voice Input
def voice_input():
    with sr.Microphone() as source:
        st.write("🎧 Speak now...")
        recognizer.adjust_for_ambient_noise(source)  # Reduce background noise
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio."
        except sr.RequestError:
            return "Speech recognition service is unavailable."
        except Exception as e:
            return f"Error: {e}"
        
        # ✅ Function: Chatbot Response


# ✅ Function: Text-to-Speech using espeak
def speak_text(text, lang="en"):
    os.system(f"espeak -v {lang} '{text}'")


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
if image:
    st.image(image, caption="📷 Uploaded Image", use_column_width=True, output_format="auto")
    extracted_text = extract_text(image)
    st.subheader("📝 Extracted Text (Translated)")
    st.write(extracted_text)

    medicine_name = identify_medicine(extracted_text)
    st.subheader("🔍 Detected Medicine")
    st.success(medicine_name)

    if medicine_name != "Unknown Medicine":
        # ✅ Step 3: Show Patient Information Form
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
        if st.button("🔎 Get Medicine Details"):
            details = get_medicine_details(medicine_name, patient_info, user_lang)
            st.subheader("📋 Medicine Details & Suitability")
            st.info(details)

            if st.button("🔊 Read Aloud", key="read_medicine"):
                os.system(f'espeak -v {user_lang} "{details}"')
                
# ✅ Inline CSS for Chatbot UI
st.markdown(
    """
    <style>
     
        .user-input {
            width: 100%;
            border: 2px solid #0d6efd;
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
            background-color: white;
            color: #333;
        }
        .chatbot-reply {
            background-color: #d1e7dd;
            color: #155724;
            padding: 15px;
            border-radius: 10px;
            font-size: 16px;
            margin-top: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .button-ask {
            background-color: #198754;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 16px;
            border: none;
            cursor: pointer;
            margin-top: 10px;
        }
        .button-ask:hover {
            background-color: #157347;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Chatbot Section
st.subheader("🤖 Chat with AI Doctor")
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

user_query = st.text_input(
    "💬 Any doubts? Chat with me:",
    placeholder="Type your question here...",
    key="chat_input"
)

if st.button("Ask Now", key="ask_now", help="Click to get an AI Doctor response"):
    if not user_query.strip():
        st.warning("⚠ Please enter a question.")
    else:
        chatbot_reply = chatbot_response(user_query, medicine_name, patient_info, user_lang)
        
        if chatbot_reply:
            st.markdown(f'<div class="chatbot-reply">🤖 AI Doctor: {chatbot_reply}</div>', unsafe_allow_html=True)
        else:
            st.error("❌ No response from AI. Please try again.")

st.markdown('</div>', unsafe_allow_html=True)  # Close chatbot container

# ✅ Project Credits
st.markdown("---")
st.markdown("### **👨‍💻 Project Developed By:**")
st.markdown(" **VIJAY, P.K. VIGNESH, and MOHAMMED FAIZAL**")

