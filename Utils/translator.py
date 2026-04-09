from langdetect import detect
from googletrans import Translator

translator = Translator()

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"


def translate_to_english(text):
    try:
        translated = translator.translate(text, dest="en")
        return translated.text
    except:
        return text


def translate_from_english(text, target_lang):
    try:
        translated = translator.translate(text, dest=target_lang)
        return translated.text
    except:
        return text