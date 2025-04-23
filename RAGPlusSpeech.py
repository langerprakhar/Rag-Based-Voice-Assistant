import os
import google.generativeai as genai
import speech_recognition as sr

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from elevenlabs import Voice, VoiceSettings, generate, play, set_api_key

# ==== CONFIGURATION ====

# 🔑 API KEYS (replace with yours)
genai.configure(api_key="GEMINI_API_KEY")
set_api_key("ELEVEN_LABS_API_KEY")

# ==== 1. Load and prepare PDF documents ====

folder_path = r"/home/prakharlanger/Rag-Based-Voice-Assistant/pdfs"
pdfs = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]

docs = []
for pdf in pdfs:
    loader = PyPDFLoader(pdf)
    docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)
vectorstore.save_local("faiss_hf_index")

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ==== 2. Set up Google Gemini ====

model = genai.GenerativeModel("gemini-1.5-flash")

# ==== 3. ElevenLabs Voice ====

def speak(text):
    print("Gemini:", text)
    try:
        audio = generate(
            text=text,
            voice="Sarah",  # You can change to other voices like "Adam", "Bella", etc.
            model="eleven_monolingual_v1"
        )
        play(audio)
    except Exception as e:
        print("Text-to-speech error:", e)

# ==== 4. Speech Recognition ====

def listen_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        print(f"You said: {command}")
        return command
    except sr.UnknownValueError:
        speak("Sorry, I couldn't understand that.")
    except sr.RequestError:
        speak("Could not connect to the speech service.")
    return ""

# ==== 5. Ask Gemini with PDF Context ====

def ask_gemini_with_context(question):
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    if not context.strip():
        prompt = question
    else:
        prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}"

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# ==== 6. Main Assistant Loop ====

if __name__ == "__main__":
    speak("Hello Prakhar! I'm Jarvis, your personalized assistant for the elderly. What would you like to know?")

    while True:
        query = listen_command()
        if query:
            if "exit" in query.lower() or "stop" in query.lower():
                speak("Goodbye!")
                break
            answer = ask_gemini_with_context(query)
            speak(answer)
