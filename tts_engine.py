import sys
import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 155)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

if __name__ == "__main__":
    text = sys.argv[1]
    speak(text)
