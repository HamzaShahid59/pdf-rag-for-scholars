from deep_translator import GoogleTranslator
from langdetect import detect  # Better language detection

def detect_and_translate(text):
    try:
        lang = detect(text)
    except Exception:
        lang = "en"

    if lang == 'en':
        return {
            "query": text,
            "translated_query": text,
            "language": "en"
        }
    else:
        translated = GoogleTranslator(source=lang, target='en').translate(text)
        return {
            "query": text,
            "translated_query": translated,
            "language": lang
        }
