import warnings
from langchain_groq import ChatGroq
import streamlit as st
from crewai import Agent, Task, Crew

# Suppress warnings
warnings.filterwarnings('ignore')

# Sidebar - Model Selection
st.sidebar.title("Settings")
model_list = [
    "distil-whisper-large-v3-en", "gemma2-9b-it", "llama-3.3-70b-versatile", 
    "llama-3.1-8b-instant", "llama-guard-3-8b", "llama3-70b-8192", 
    "llama3-8b-8192", "mixtral-8x7b-32768", "whisper-large-v3", "whisper-large-v3-turbo"
]
model_choice = st.sidebar.selectbox("Choose LLM Model:", model_list)
api_key = st.sidebar.text_input("Enter API Key:", type="password")

# Sidebar - Blog Settings
topic = st.sidebar.text_input("Enter a topic:")

if api_key:
    llm = ChatGroq(
        temperature=0,
        model_name=model_choice,
        api_key=api_key
    )

    # Define Agents
    planner = Agent(
        llm=llm,
        role="Content Strategist",
        goal=f"Curate and structure an engaging and factually accurate blog on {topic}",
        backstory="You analyze key insights, trends, and audience interests to create a compelling content strategy.",
        allow_delegation=False,
        verbose=True
    )

    writer = Agent(
        llm=llm,
        role="Expert Content Writer",
        goal=f"Write an insightful and well-researched article on {topic}",
        backstory="You transform structured content into a compelling, informative, and engaging blog post tailored for readers.",
        allow_delegation=False,
        verbose=True
    )

    editor = Agent(
        llm=llm,
        role="Senior Editor",
        goal="Refine and polish the blog post for clarity, coherence, and readability.",
        backstory="You ensure high-quality, error-free content while enhancing engagement and SEO compliance.",
        allow_delegation=False,
        verbose=True
    )

    # Define Tasks
    plan = Task(
        description=f"1. Research the latest trends and critical details about {topic}.",
        expected_output="A structured content outline with an audience profile, key topics, and SEO-optimized keywords.",
        agent=planner,
    )

    write = Task(
        description=f"1. Develop an engaging blog post following the content outline.\n"
                    "2. Organize it with clear headings, compelling subheadings, and structured sections.\n"
                    "3. Maintain readability, optimize for SEO, and ensure user engagement.",
        expected_output="A polished, well-structured blog post ready for publication.",
        agent=writer,
    )

    edit = Task(
        description="Review, proofread, and enhance the blog post, ensuring factual accuracy, readability, and SEO effectiveness.",
        expected_output="A final refined article meeting professional editorial standards.",
        agent=editor,
    )

    # Define Crew
    crew = Crew(
        agents=[planner, writer, editor],
        tasks=[plan, write, edit],
        verbose=True
    )

    def generate_content(topic):
        return crew.kickoff(inputs={"topic": topic})

    # Main UI
    st.title("AI-Powered Blog Generator")
    if st.sidebar.button("Generate Blog Post"):
        if topic:
            with st.spinner("Generating content... Please wait!"):
                output = generate_content(topic)
            st.markdown(output)
            st.download_button("Download Blog", output, file_name=f"{topic.replace(' ', '_')}.md")
        else:
            st.error("Please enter a topic before generating.")
