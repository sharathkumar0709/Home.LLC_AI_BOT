import speech_recognition as sr
import pyttsx3
from fuzzywuzzy import process
from transformers import pipeline

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Adjust speed

# Custom predefined responses (your answers)
faq = {
    "what is your name": "I am your personal AI assistant.",
    "how are you": "I am just a program, but I'm always here to help!",
    "What is your first superpower":"My #1 superpower is problem-solving—I thrive on tackling complex AI/ML challenges, breaking them down into actionable steps, and delivering scalable solutions. Whether it’s optimizing an ML model, automating workflows, or fine-tuning an LLM, I always find a way to improve performance and efficiency.",
    "What are the top 3 areas you would like to grow in":"I want to deepen my expertise in Advanced LLM Fine-tuning & Optimization, focusing on training and optimizing models like LLaMA, Mistral, and GPT-4. Additionally, I aim to enhance my skills in MLOps & Scalable AI Deployments, including CI/CD for AI, model monitoring, and cloud-based ML pipelines. Lastly, I’m eager to explore AI & Blockchain Integration, leveraging decentralized AI and smart contracts to improve AI security and model sharing.",
    "What misconception do your coworkers have about you": "Some may think I’m too focused on technical work, but in reality, I enjoy collaborating with teams, mentoring peers, and translating AI insights into business value.",
    "How do you push your boundaries and limits":"I constantly challenge myself by learning new technologies, working on side projects, and taking on complex AI tasks outside my comfort zone. I also engage with the AI research community, experiment with cutting-edge models, and stay updated with industry advancements to keep growing."
}


# Load a free AI chatbot model (works offline after download)
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

def speak(text):
    """Converts text to speech."""
    engine.say(text)
    engine.runAndWait()

def listen():
    """Listens for user input and converts speech to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio).lower()
        print(f"You: {text}")
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand that."
    except sr.RequestError:
        return "Speech recognition service unavailable."

def find_best_answer(question):
    """Finds the best matching question in the database and returns the answer."""
    best_match, score = process.extractOne(question, faq.keys())  # Find best match
    if score > 60:  # If match is strong enough
        return faq[best_match]
    else:
        return None  # No predefined answer found

def generate_ai_response(prompt):
    """Generates an AI response if no predefined answer exists."""
    response = chatbot(prompt, max_length=100, do_sample=True, pad_token_id=50256, truncation=True)
    return response[0]["generated_text"]

# Run the chatbot
print("Say 'exit' to stop.")
while True:
    user_input = listen()
    if "exit" in user_input:
        speak("Goodbye!")
        break
    
    # Check predefined responses first
    response = find_best_answer(user_input)
    
    if response is None:  # If no predefined answer, generate AI response
        response = generate_ai_response(user_input)
    
    print(f"Chatbot: {response}")
    speak(response)
