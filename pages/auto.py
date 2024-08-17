import streamlit as st
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

# Load environment variables from .env file
load_dotenv()

# Add explanation at the top of the app
st.write("""
This AI Project Generator helps you create a comprehensive project outline and implementation details for your AI product. 
To use it:
1. Enter your AI Product name
2. Specify the AI Technology, Language, or Framework you want to use
3. List the Key Features of your project
4. Click 'Generate Project' to get detailed project information, including folder structure, implementation plan, and sample code.
""")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Initialize the OpenAI language model
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.7,
    max_tokens=16384,
    streaming=True
)

# Define the project generation prompt
project_prompt = PromptTemplate(
    input_variables=["AI_PRODUCT", "AI_TECH_LANG_FRAME", "KEY_FEATURES"],
    template="""
Generate a comprehensive project outline and implementation details for an {AI_PRODUCT} using {AI_TECH_LANG_FRAME}. The core features for this Proof of Concept (POC) are: {KEY_FEATURES}

Please provide:
1. A detailed folder and file structure following clean architecture principles.
2. An explanation of the main files and their purposes.
3. A comprehensive implementation plan, detailing each step of the development process.
4. Best practices and potential challenges for this type of project.
5. A bash script to create the folder and file structure.
6. Sample code for key components (e.g., main application file, crucial models, important functions).
7. A requirements.txt file listing all necessary dependencies.

Ensure that the output is detailed enough to serve as a complete guide for implementing the project.
"""
)

# Create LangChain
project_chain = LLMChain(llm=llm, prompt=project_prompt)

# Streamlit app
st.title("AI Project Generator")

# Project inputs
ai_product = st.text_input("AI Product")
ai_tech_lang_frame = st.text_input("AI Technology/Language/Framework")
key_features = st.text_area("Key Features")

if st.button("Generate Project"):
    if ai_product and ai_tech_lang_frame and key_features:
        st.subheader("Generated Project Details")
        output_container = st.empty()
        stream_handler = StreamHandler(output_container)
        
        # Create LangChain with streaming
        project_chain = LLMChain(llm=llm, prompt=project_prompt)
        
        response = project_chain.run(
            {
                "AI_PRODUCT": ai_product,
                "AI_TECH_LANG_FRAME": ai_tech_lang_frame,
                "KEY_FEATURES": key_features
            },
            callbacks=[stream_handler]
        )
        
        # Extract and display the bash script
        bash_script_start = response.find("```bash")
        bash_script_end = response.find("```", bash_script_start + 7)
        if bash_script_start != -1 and bash_script_end != -1:
            bash_script = response[bash_script_start+7:bash_script_end].strip()
            st.subheader("Bash Script for Project Structure")
            st.code(bash_script, language="bash")
        
        # Option to download the project details
        st.download_button(
            label="Download Project Details",
            data=response,
            file_name="project_details.txt",
            mime="text/plain"
        )
    else:
        st.warning("Please fill in all fields.")