import os
import streamlit as st
import pandas as pd
from openai import OpenAI
import re
import requests
from docx import Document
import io
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Jona's Utilities")

st.title("Jona's Utilities")
st.write("Here are a list of useful utilities I've built\n\n")
st.page_link("pages/requirements.py",  label="PRD generator", icon="🔎")
st.page_link("pages/auto.py",  label="Auto Project setup", icon="📂")