from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langdetect import detect
from transformers import pipeline, MarianMTModel, MarianTokenizer

# Sample bilingual proverbs dataset
proverbs = [
    {"text": "الصبر مفتاح الفرج", "meaning": "Patience is the key to relief.", "lang": "ar"},
    {"text": "A friend in need is a friend indeed.", "meaning": "True friends show themselves during tough times.", "lang": "en"},
    {"text": "يد واحدة لا تصفق", "meaning": "One hand cannot clap.", "lang": "ar"},
    {"text": "Don’t count your chickens before they hatch.", "meaning": "Don’t assume future success before it happens.", "lang": "en"},
]

# Load multilingual embedding model
print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Prepare embeddings
texts = [p["text"] for p in proverbs]
embeddings = model.encode(texts)
embeddings = np.array(embeddings).astype("float32")

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Load summarizer (can also be used for poetic rephrasing)
print("Loading summarization model...")
summarizer = pipeline("text2text-generation", model="google/flan-t5-small")

# Load translation model and tokenizer (MarianMT)
print("Loading translation model...")
translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
translation_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")

# Translation function
def translate_text(text, src_lang, tgt_lang):
    if src_lang == 'en' and tgt_lang == 'ar':
        translated = translation_model.generate(**translation_tokenizer(text, return_tensors="pt", padding=True))
        return translation_tokenizer.decode(translated[0], skip_special_tokens=True)
    elif src_lang == 'ar' and tgt_lang == 'en':
        translated = translation_model.generate(**translation_tokenizer(text, return_tensors="pt", padding=True))
        return translation_tokenizer.decode(translated[0], skip_special_tokens=True)
    else:
        return text  # Return the original text if no translation needed

# Ask user for input
print("\nWelcome to Tarjamat Al-Turath!")
query = input("Ask about a proverb or enter a phrase: ")

# Detect language
lang = detect(query)

# Encode user query and search
query_vec = model.encode([query]).astype("float32")
D, I = index.search(query_vec, k=1)
match = proverbs[I[0][0]]

# Display result
print("\n🔎 Best Match:")
print(f"📜 Proverb: {match['text']}")
print(f"💡 Meaning: {match['meaning']}")

# Rephrase it creatively
rephrase = summarizer(f"Rephrase this proverb poetically: {match['meaning']}", max_length=50)[0]["generated_text"]
print(f"🎨 Poetic Rephrase: {rephrase}")

# Translate if needed
if lang != match["lang"]:
    print(f"\n🌐 Translation for your convenience:")
    if lang == 'en':
        translated = translate_text(match['meaning'], 'ar', 'en')
        print(f"Arabic ➜ English: {translated}")
    elif lang == 'ar':
        translated = translate_text(match['meaning'], 'en', 'ar')
        print(f"English ➜ Arabic: {translated}")
