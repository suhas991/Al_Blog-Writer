import warnings
from langchain_groq import ChatGroq
import streamlit as st
from crewai import Agent, Task, Crew


# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize LLM
# BASE_MODEL = "deepseek-r1"
# OLLAMA_API = "http://localhost:11434"
# llm = Ollama(model=BASE_MODEL, base_url=OLLAMA_API)

llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    api_key="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"
)
# Define Agents
planner = Agent(
    llm=llm,
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="You collect and organize information for an informative blog post.",
    allow_delegation=False,
    verbose=True
)

writer = Agent(
    llm=llm,
    role="Content Writer",
    goal="Write an insightful blog post on {topic}",
    backstory="You create a well-structured article based on the Content Planner's input.",
    allow_delegation=False,
    verbose=True
)

editor = Agent(
    llm=llm,
    role="Editor",
    goal="Proofread and refine the blog post for clarity and accuracy.",
    backstory="You ensure the content meets high editorial standards.",
    allow_delegation=False,
    verbose=True
)

# Define Tasks
plan = Task(
    description=(
        "1. Gather latest trends and key details about {topic}.\n"
        "2. Identify target audience and their needs.\n"
        "3. Develop an article outline with SEO keywords."
    ),
    expected_output="A content plan with an outline and keyword suggestions.",
    agent=planner,
)

write = Task(
    description=(
        "1. Write an engaging blog post based on the plan.\n"
        "2. Structure it well with headings and key points.\n"
        "3. Ensure clarity, SEO optimization, and readability."
    ),
    expected_output="A structured blog post in markdown format.",
    agent=writer,
)

edit = Task(
    description="Proofread and enhance the blog post.",
    expected_output="A refined blog post ready for publishing.",
    agent=editor,
)

# Define Crew
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=True
)

def generate_content(topic):
    result = crew.kickoff(inputs={"topic": topic})
    return result

# Streamlit UI
st.title("AI-Powered Blog Generator")
topic = st.text_input("Enter a topic:")
if st.button("Generate Blog Post"):
    if topic:
        st.write("Generating content... Please wait!")
        output = generate_content(topic)
        st.markdown(output)
    else:
        st.error("Please enter a topic before generating.")
