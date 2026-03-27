import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.title("Sourav Academic Rewrite Engine (PRO)")

st.write("1500+ words supported | No summarization | Structured rewrite")

@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# 🔹 SPLIT TEXT INTO CHUNKS
def split_text(text, max_words=400):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
        
    return chunks

# 🔹 REWRITE EACH CHUNK
def rewrite_chunk(text):
    prompt = f"""
Rewrite the following text in a formal academic research style.

IMPORTANT:
- DO NOT summarize
- DO NOT shorten
- Preserve full meaning
- Maintain sentence count
- Rewrite sentence-by-sentence

Text:
{text}
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    outputs = model.generate(
        **inputs,
        max_new_tokens=600,
        min_length=300,
        num_beams=5,
        length_penalty=1.0,
        no_repeat_ngram_size=3
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 🔹 MAIN FUNCTION
def rewrite_long(text):
    chunks = split_text(text)
    rewritten_chunks = []
    
    for chunk in chunks:
        rewritten = rewrite_chunk(chunk)
        rewritten_chunks.append(rewritten)
    
    return "\n\n".join(rewritten_chunks)

# UI
input_text = st.text_area("Paste your text (1500–3000 words supported)", height=300)

if st.button("Rewrite"):
    if input_text.strip() == "":
        st.warning("Please enter text")
    else:
        with st.spinner("Processing large text..."):
            output = rewrite_long(input_text)
        st.subheader("Rewritten Output:")
        st.write(output)
