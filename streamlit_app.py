import streamlit as st
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langdetect import detect
from transformers import pipeline, MarianMTModel, MarianTokenizer

# Load data
with open("data/proverbs.json", "r", encoding="utf-8") as f:
    proverbs = json.load(f)

texts = [p["text"] for p in proverbs]

# Load models
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    summarizer = pipeline("text2text-generation", model="csebuetnlp/mT5_multilingual_XLSum")
    translator = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    return embedder, summarizer, translator, tokenizer

embedder, summarizer, translator_model, tokenizer = load_models()

# Create FAISS index
embeddings = embedder.encode(texts).astype("float32")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Translation function
def translate(text, src_lang, tgt_lang):
    if src_lang == tgt_lang:
        return text
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    outputs = translator_model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# UI
st.title("🕌 Tarjamat Al-Turath")
st.subheader("Discover and Translate Arabic & English Proverbs")

query = st.text_input("🔍 Enter your query (in Arabic or English):")

if query:
    lang = detect(query)
    query_vec = embedder.encode([query]).astype("float32")
    D, I = index.search(query_vec, k=3)  # Top 3 matches

    st.markdown("### 📜 Results")
    for idx in I[0]:
        result = proverbs[idx]
        st.markdown(f"**🧾 Proverb:** {result['text']}")
        st.markdown(f"**💡 Meaning:** {result['meaning']}")

        poetic = summarizer(f"Rephrase poetically: {result['meaning']}", max_length=50)[0]["generated_text"]
        st.markdown(f"**🎨 Poetic Rephrase:** _{poetic}_")

        # Show translation if language is different
        if lang != result["lang"]:
            translated = translate(result["meaning"], result["lang"], lang)
            st.markdown(f"**🌐 Translated Meaning:** {translated}")

        st.markdown("---")
