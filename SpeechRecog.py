import speech_recognition as sr
import serial
import pyttsx3
import pyaudio

arduino = serial.Serial(port="COM3", baudrate=9600, timeout=.1)

recognizer = sr.Recognizer()
relayNumber = 0

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

            text = text.split()
            print(text)

            if 'relay' in text:
                print(relayNumber)

            if relayNumber == 0:
                if 'one' in text or 'won' in text or '1' in text:
                    relayNumber = 1
                    arduino.write(b'2')
                    print("MODE 1")
                elif 'two' in text or 'too' in text or 'to' in text or '2' in text:
                    relayNumber = 2
                    arduino.write(b'4')
                    print("MODE 2")
            elif relayNumber == 1:
                if 'on' in text:
                    arduino.write(b'1')
                elif 'off' in text or 'of' in text:
                    arduino.write(b'0')
                elif 'one' in text or 'won' in text:
                    relayNumber = 0
                    arduino.write(b'2')
                    print("Mode 1 out")
            elif relayNumber == 2:
                if 'on' in text:
                    arduino.write(b'1')
                elif 'of' in text or 'off' in text:
                    arduino.write(b'0')
                elif 'two' in text or 'too' in text or 'to' in text or '2' in text:
                    relayNumber = 0
                    arduino.write(b'2')
                    print("Mode 2 out")

    except:
        print("Unable to convert")
        continue
