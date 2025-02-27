import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

# Ensure the trained model exists
MODEL_PATH = "trained_transformer"
if not os.path.exists(MODEL_PATH):
    st.error("❌ Trained model not found! Please upload the fine-tuned model or retrain it.")
    st.stop()
else:
    st.write("✅ Trained model found. Loading...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    model.eval()

def generate_cpp_code(pseudocode):
    input_text = "translate pseudocode to C++: " + pseudocode
    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"],
            max_length=256,
            num_return_sequences=1,
            temperature=0.7,  # Adjust for better randomness
            top_k=50,  # Reduce output repetition
            top_p=0.95  # Nucleus sampling
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
st.title("Pseudocode to C++ Code Generator")
st.write("Enter pseudocode below, and click 'Generate Code' to convert it into C++.")

user_input = st.text_area("Enter Pseudocode:")
if st.button("Generate Code"):
    if user_input.strip():
        generated_code = generate_cpp_code(user_input)
        st.subheader("Generated C++ Code:")
        st.code(generated_code, language='cpp')
    else:
        st.warning("⚠️ Please enter some pseudocode before generating.")