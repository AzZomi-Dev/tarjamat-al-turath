import streamlit as st
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langdetect import detect
from transformers import pipeline, MarianMTModel, MarianTokenizer

# 🌟 Page setup
st.set_page_config(
    page_title="ترجمة التراث | Tarjamat Al-Turath",
    page_icon="🕌",
    layout="centered",
)

# 🧠 Language toggle
lang = st.radio("🌐 اختر اللغة | Select Language", ["العربية", "English"], horizontal=True, index=0)

# 📜 App Description
if lang == "العربية":
    st.title("🕌 ترجمة التراث")
    st.markdown("""
    ✨ استكشف الأمثال العربية بطريقة جديدة — باستخدام الذكاء الاصطناعي لترجمتها، وتفسيرها، وإعادة صياغتها بأسلوب شعري.
    
    📚 أدخل مثلًا أو عبارة، وسنقوم بتحليلها، ترجمتها، وإظهار معناها.
    """)
else:
    st.title("🕌 Tarjamat Al-Turath")
    st.markdown("""
    ✨ Discover Arabic proverbs in a whole new way — with AI-powered translation, interpretation, and poetic rephrasing.
    
    📚 Enter a proverb or phrase, and we’ll break it down, translate it, and offer its meaning.
    """)

# Load data
with open("data/proverbs.json", "r", encoding="utf-8") as f:
    proverbs = json.load(f)

texts = [p["text"] for p in proverbs]

# Load models

@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    summarizer = pipeline("text2text-generation", model="google/flan-t5-small")  # ✅ Streamlit-safe
    translator_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    return embedder, summarizer, translator_model, tokenizer

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
st.subheader("Discover and Translate Arabic & English Proverbs")

# Input field based on selected UI language
if lang == "العربية":
    query = st.text_input("🔍 أدخل سؤالك أو مثلك (بالعربية أو الإنجليزية):", key="query_arabic")
else:
    query = st.text_input("🔍 Enter your query (in Arabic or English):", key="query_english")

if query:
    lang_detected = detect(query)  # Detect language of the query
    query_vec = embedder.encode([query]).astype("float32")
    D, I = index.search(query_vec, k=3)  # Top 3 matches

# Only continue if query exists
if query.strip():
    query_lang = detect(query)  # Detect the language of the user query
    query_vec = embedder.encode([query]).astype("float32")
    D, I = index.search(query_vec, k=3)  # Top 3 matches

    # Header based on language
    if lang == "العربية":
        st.markdown("### 📜 النتائج")
    else:
        st.markdown("### 📜 Results")

    for idx in I[0]:
        result = proverbs[idx]

        # Display results based on language
        if lang == "العربية":
            st.markdown(f"**🧾 المثل:** {result['text']}")
            st.markdown(f"**💡 المعنى:** {result['meaning']}")
        else:
            st.markdown(f"**🧾 Proverb:** {result['text']}")
            st.markdown(f"**💡 Meaning:** {result['meaning']}")

        # Generate the poetic rephrase with creativity
        poetic = summarizer(f"Create a poetic and elegant version of this meaning: {result['meaning']}", max_length=100)[0]["generated_text"]

        if lang == "العربية":
            st.markdown(f"**🎨 إعادة الصياغة الشعرية:** _{poetic}_")
        else:
            st.markdown(f"**🎨 Poetic Rephrase:** _{poetic}_")

        # Translate meaning if needed
        if query_lang != result["lang"]:
            translated = translate(result["meaning"], result["lang"], query_lang)
            if lang == "العربية":
                st.markdown(f"**🌐 الترجمة:** {translated}")
            else:
                st.markdown(f"**🌐 Translated Meaning:** {translated}")

        st.markdown("---")
