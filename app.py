import streamlit as st
from transformers import pipeline

st.title("Sourav Academic Rewrite Engine")

st.write("Rewrite AI text into structured academic style")

@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

generator = load_model()

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
    result = generator(prompt, max_length=512, do_sample=False)
    return result[0]['generated_text']

input_text = st.text_area("Paste your text (up to ~800–1000 words)", height=250)

if st.button("Rewrite"):
    if input_text.strip() == "":
        st.warning("Please enter text")
    else:
        with st.spinner("Rewriting..."):
            output = rewrite(input_text)
        st.subheader("Output:")
        st.write(output)
