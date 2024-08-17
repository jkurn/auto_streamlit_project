import streamlit as st
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler

# Load environment variables from .env file
load_dotenv()

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

# Update the PRD generation prompt
prd_prompt = PromptTemplate(
    input_variables=[
        "product_description", "target_audience", "problem_statement",
        "unique_value_prop", "differentiation", "monetization", "acquisition_channel"
    ],
    template="""
You are an expert writer of Product Requirements Documents (PRDs). Your task is to draft a comprehensive PRD based on the information provided and make targeted assumptions where necessary. The goal is to create a document that leaves no questions unanswered for designers and engineers.

Here's the template structure you should follow for the PRD:

1. Problem
2. High Level Approach
3. Narrative
4. Goals
   4.1 Metrics
   4.2 Impact Sizing Model
5. Non-goals
6. Solution Alignment
7. Key Features
   7.1 Plan of record
   7.2 Future considerations
8. Key Flows
9. Key Logic
10. Launch Plan
11. Key Milestones

Use the following information as the foundation for your PRD:

1. Product & Description:
{product_description}

2. Target Audience:
{target_audience}

3. Problem Statement, Goal, and Motivation:
{problem_statement}

4. Unique Value Proposition & Benefit to Key Users' Pain Point:
{unique_value_prop}

5. Differentiation and Alternatives:
{differentiation}

6. Monetization, Willingness to Pay, Friction:
{monetization}

7. Acquisition Channel:
{acquisition_channel}

Where the provided information is incomplete, make informed assumptions based on industry best practices and your expertise. Ensure these assumptions align with the overall product strategy and user needs.

For each section of the PRD:
1. Start with the known information provided.
2. Expand on this information using your expertise and reasonable assumptions.
3. Ensure each section is detailed and leaves no room for ambiguity.
4. Use clear, concise language that both technical and non-technical stakeholders can understand.

When drafting the PRD:
- In the Problem section, clearly articulate the user pain point and business opportunity.
- For the High Level Approach, outline a strategic plan that addresses the problem effectively.
- In the Narrative, create compelling user stories that cover both common and edge cases.
- For Goals and Metrics, set ambitious yet achievable targets. Include specific numbers where possible.
- In the Impact Sizing Model, show your calculations and reasoning clearly.
- For Non-goals, be explicit about what's out of scope and why.
- In the Solution Alignment and Key Features sections, be specific about what will be built.
- For Key Flows and Key Logic, provide detailed step-by-step descriptions.
- In the Launch Plan, create a realistic timeline with clear phase definitions.
- For Key Milestones, include specific dates or timeframes where possible.

Format your PRD using markdown for readability. Use headers, bullet points, and tables where appropriate.

After completing the PRD draft, identify any areas where you made significant assumptions or where more information would be beneficial. List these as follow-up questions at the end of your document.

Present your final PRD draft within <PRD> tags, and list your follow-up questions within <QUESTIONS> tags.

Remember to approach this task with confidence, demonstrating strong strategic thinking, UX sensibility, and business acumen throughout the document.
"""
)

# Create LangChain
prd_chain = LLMChain(llm=llm, prompt=prd_prompt)

# Streamlit app
st.title("PRD Generator")
st.write("""
This tool helps you create a comprehensive Product Requirements Document (PRD) based on your product strategy hypothesis. 
Fill in the fields below with your product details, and click 'Generate PRD' to get a well-structured document.

For each field, we're looking for specific information about your product strategy:

1. Product & Description: Enter your product name and a brief explanation of what it does.
2. Target Audience: Describe your primary market segment and any secondary users you might explore in the future.
3. Problem Statement, Goal, and Motivation: Explain the end outcome users want to achieve, why they want it, and the current pain points or unmet needs.
4. Unique Value Proposition: Describe how your product solves key user pain points, include a one-liner tagline, and any potential user skepticism.
5. Differentiation and Alternatives: List current solutions or workarounds, their drawbacks, and your product's unique differentiators.
6. Monetization Strategy: Identify the decision-maker, price point, reasons for willingness to pay, related past purchases, and potential buying frictions.
7. Acquisition Channel: Describe where your target users spend their time (online or offline) that you can reach them.

If you're unsure about any field, you can enter '[TBD]' and revisit it later.
""")

# New input fields with more detailed descriptions
product_description = st.text_input("1. Product & Description", 
                                    help="Enter your product name and a brief explanation of what it does.")
target_audience = st.text_input("2. Target Audience", 
                                help="Describe your primary market segment and any secondary users you might explore in the future.")
problem_statement = st.text_input("3. Problem Statement, Goal, and Motivation", 
                                  help="Explain the end outcome users want to achieve, why they want it, and the current pain points or unmet needs.")
unique_value_prop = st.text_input("4. Unique Value Proposition", 
                                  help="Describe how your product solves key user pain points, include a one-liner tagline, and any potential user skepticism.")
differentiation = st.text_input("5. Differentiation and Alternatives", 
                                help="List current solutions or workarounds, their drawbacks, and your product's unique differentiators.")
monetization = st.text_input("6. Monetization Strategy", 
                             help="Identify the decision-maker, price point, reasons for willingness to pay, related past purchases, and potential buying frictions.")
acquisition_channel = st.text_input("7. Acquisition Channel", 
                                    help="Describe where your target users spend their time (online or offline) that you can reach them.")

if st.button("Generate PRD"):
    if all([product_description, target_audience, problem_statement, unique_value_prop, differentiation, monetization, acquisition_channel]):
        st.subheader("Generated PRD")
        output_container = st.empty()
        stream_handler = StreamHandler(output_container)
        
        # Create LangChain with streaming
        prd_chain = LLMChain(llm=llm, prompt=prd_prompt)
        
        response = prd_chain.run(
            {
                "product_description": product_description,
                "target_audience": target_audience,
                "problem_statement": problem_statement,
                "unique_value_prop": unique_value_prop,
                "differentiation": differentiation,
                "monetization": monetization,
                "acquisition_channel": acquisition_channel
            },
            callbacks=[stream_handler]
        )
        
        # Extract PRD and questions
        prd_start = response.find("<PRD>")
        prd_end = response.find("</PRD>")
        questions_start = response.find("<QUESTIONS>")
        questions_end = response.find("</QUESTIONS>")
        
        if prd_start != -1 and prd_end != -1:
            prd_content = response[prd_start+5:prd_end].strip()
            # Remove the streaming output
            output_container.empty()
            # Display the final PRD content
            st.markdown(prd_content)
        
        if questions_start != -1 and questions_end != -1:
            questions_content = response[questions_start+11:questions_end].strip()
            st.subheader("Follow-up Questions")
            st.markdown(questions_content)
        
        # Option to download the PRD
        st.download_button(
            label="Download PRD",
            data=response,
            file_name="product_requirements_document.md",
            mime="text/markdown"
        )
    else:
        st.warning("Please fill in all the fields.")