import speech_recognition as sr
import pyttsx3
import pyaudio

recognizer = sr.Recognizer()

while True:
    try:
        with sr.Microphone() as mic:
            recognizer.adjust_for_ambient_noise(mic, duration=0.2)

            print("\nListening...")
            audio = recognizer.listen(mic)

            print("\nRecognizing Audio...")
            text = recognizer.recognize_google(audio)
            text = text.lower()

            print(text)
            if text == "exit":
                break

    except sr.UnknownValueError():
        recognizer = sr.Recognizer()
        continue