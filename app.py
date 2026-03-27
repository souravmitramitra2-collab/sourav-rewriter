import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.title("Sourav Academic Rewrite Engine")

st.write("Rewrite AI text into structured academic style")

@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

def rewrite(text):
    prompt = f"""
Rewrite the following text in a formal academic research style.

Rules:
- Preserve meaning
- Improve clarity
- No expansion
- No shortening
- Maintain structured academic tone

Text:
{text}
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

input_text = st.text_area("Paste your text (up to ~800 words)", height=250)

if st.button("Rewrite"):
    if input_text.strip() == "":
        st.warning("Please enter text")
    else:
        with st.spinner("Rewriting..."):
            output = rewrite(input_text)
        st.subheader("Output:")
        st.write(output)
