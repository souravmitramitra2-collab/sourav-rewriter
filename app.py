import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from docx import Document
import io

st.title("Sourav Academic Engine v8")

st.write("Style-trained | Structure preserved | DOCX export ready")

@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# 🔹 STYLE PHRASE MEMORY
STYLE_PHRASES = [
    "It is observed that",
    "It can be observed that",
    "From the observations",
    "This shows that",
    "An attempt is made",
    "The study indicates",
    "Further analysis shows"
]

# 🔹 DETECT BULLETS
def is_bullet(line):
    return line.strip().startswith(("•", "-", "*", "r", "+"))

# 🔹 DETECT HEADINGS
def is_heading(line):
    return line.strip().isupper() or len(line.split()) < 6

# 🔹 BULLET → PARAGRAPH
def convert_bullets(lines):
    paragraph = " ".join(lines)
    return paragraph

# 🔹 REWRITE FUNCTION
def rewrite_paragraph(text):

    prompt = f"""
Rewrite in structured academic style using these characteristics:

- Use analytical tone
- Maintain sentence depth
- Use phrases like: {", ".join(STYLE_PHRASES)}
- Maintain logical flow
- Do NOT summarize
- Preserve technical meaning

Text:
{text}
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    outputs = model.generate(
        **inputs,
        max_new_tokens=700,
        min_length=250,
        num_beams=5,
        length_penalty=1.1,
        no_repeat_ngram_size=3
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 🔹 MAIN ENGINE
def process_text(text):
    lines = text.split("\n")
    output = []
    bullet_group = []

    for line in lines:
        if is_bullet(line):
            bullet_group.append(line)
        else:
            if bullet_group:
                para = convert_bullets(bullet_group)
                output.append(rewrite_paragraph(para))
                bullet_group = []

            if is_heading(line):
                output.append(line)
            elif line.strip() == "":
                output.append("")
            else:
                output.append(rewrite_paragraph(line))

    return "\n".join(output)

# 🔹 DOCX EXPORT
def create_docx(text):
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# UI
input_text = st.text_area("Paste your structured research text", height=300)

if st.button("Rewrite"):
    if input_text.strip() == "":
        st.warning("Please enter text")
    else:
        with st.spinner("Processing with Style Engine v8..."):
            output = process_text(input_text)
        st.subheader("Rewritten Output:")
        st.text(output)

        # Download button
        docx_file = create_docx(output)
        st.download_button(
            label="Download as DOCX",
            data=docx_file,
            file_name="rewritten_paper.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
