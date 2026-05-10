import os
import time
from gtts import gTTS

SOUNDS_DIR = "assets/sounds"

# maps project key -> Arabic pronunciation with tashkeel
arabic_letters = {
    "aleff": "أَلِف",
    "bb":    "بَا",
    "ta":    "تَا",
    "thaa":  "ثَاء",
    "jeem":  "جِيم",
    "haa":   "حَا",
    "khaa":  "خَا",
    "dal":   "دَال",
    "thal":  "ذَال",
    "ra":    "رَا",
    "zay":   "زَاي",
    "seen":  "سِين",
    "sheen": "شِين",
    "saad":  "صَاد",
    "dhad":  "ضَاد",
    "taa":   "طَا",
    "dha":   "ظَا",
    "ain":   "عَين",
    "ghain": "غَين",
    "fa":    "فَا",
    "gaaf":  "قَاف",
    "kaaf":  "كَاف",
    "laam":  "لَام",
    "meem":  "مِيم",
    "nun":   "نُون",
    "ha":    "هَا",
    "waw":   "وَاو",
    "ya":    "يَا",
    "al":    "أَل",
    "yaa":   "أَلِف مَقْصُورَة",
    "toot":  "تَا مَرْبُوطَة",
}

os.makedirs(SOUNDS_DIR, exist_ok=True)

print("Generating MP3 files...")

for key, arabic_text in arabic_letters.items():
    try:
        tts = gTTS(text=arabic_text, lang='ar')
        file_path = os.path.join(SOUNDS_DIR, f"{key}_arsl.mp3")
        tts.save(file_path)
        print(f"Saved: {file_path}")
        time.sleep(0.5)
    except Exception as e:
        print(f"Error generating {key}: {e}")

print(f"\nDone! All MP3s are saved in the '{SOUNDS_DIR}' directory.")
