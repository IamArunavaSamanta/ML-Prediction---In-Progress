
import streamlit as st

st.title("Hello Streamlit in Jupyter!")
st.write("This is a simple Streamlit app running from a Jupyter Notebook.")
name = st.text_input("Enter your name:")
if name:
    st.success(f"Hello, {name}!")
