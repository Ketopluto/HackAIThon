import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import json
from typing import List, Dict

# Load environment variables
load_dotenv()

# Modified prompt templates with stricter JSON formatting
PREREQUISITES_TEMPLATE = """Generate a list of essential prerequisites for the topic '{topic}'.
IMPORTANT: Respond with ONLY a JSON array of strings, like this:
["prerequisite 1", "prerequisite 2", "prerequisite 3"]"""

SUBTOPICS_TEMPLATE = """Based on the topic '{topic}' and the user's proficiency levels in prerequisites:
{prereq_proficiency}

Generate relevant subtopics for their level of understanding.
IMPORTANT: Respond with ONLY a JSON array of strings, like this:
["subtopic 1", "subtopic 2", "subtopic 3"]"""

ROADMAP_TEMPLATE = """Create a learning roadmap for {topic} focusing on these subtopics:
{selected_subtopics}

Consider the user's prerequisite knowledge:
{prereq_proficiency}

IMPORTANT: Format your response as a valid JSON object like this example:
{{
    "sequence": ["step 1", "step 2", "step 3"],
    "pitfalls": ["pitfall 1", "pitfall 2"],
    "focus_areas": ["area 1", "area 2"],
    "time_estimate": "Estimated time description"
}}
Ensure to use these exact key names and format."""

CHATBOT_TEMPLATE = """You are an educational assistant helping students understand {topic}.
Chat History: {chat_history}
Student's Question: {human_input}

Provide a helpful response considering:
- Prerequisites knowledge: {prereq_proficiency}
- Selected subtopics: {selected_subtopics}"""

def clean_json_string(s: str) -> str:
    """Clean and extract JSON string from response."""
    # Try to find JSON content between triple backticks if present
    if "```json" in s:
        s = s.split("```json")[1].split("```")[0]
    elif "```" in s:
        s = s.split("```")[1].split("```")[0]
    return s.strip()

def parse_json_response(response_content: str, default_value: any) -> any:
    """Safely parse JSON response with fallback to default value."""
    try:
        cleaned_content = clean_json_string(response_content)
        return json.loads(cleaned_content)
    except (json.JSONDecodeError, IndexError) as e:
        st.error(f"Error parsing response: {str(e)}")
        st.write("Raw response:", response_content)
        return default_value

def initialize_llm():
    """Initialize the Groq LLM with specified parameters."""
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192",
        temperature=0.7,
        max_tokens=4096,
    )

def get_prerequisites(llm, topic: str) -> List[str]:
    """Generate prerequisites for the given topic."""
    prompt = PromptTemplate(
        input_variables=["topic"],
        template=PREREQUISITES_TEMPLATE
    )
    response = llm.invoke(prompt.format(topic=topic))
    return parse_json_response(response.content, ["Basic understanding of the subject"])

def get_subtopics(llm, topic: str, prereq_proficiency: Dict[str, str]) -> List[str]:
    """Generate subtopics based on topic and prerequisite proficiency."""
    prompt = PromptTemplate(
        input_variables=["topic", "prereq_proficiency"],
        template=SUBTOPICS_TEMPLATE
    )
    response = llm.invoke(prompt.format(
        topic=topic,
        prereq_proficiency=json.dumps(prereq_proficiency)
    ))
    return parse_json_response(response.content, ["Introduction to " + topic])

def generate_roadmap(llm, topic: str, selected_subtopics: List[str], prereq_proficiency: Dict[str, str]) -> Dict:
    """Generate a learning roadmap based on selected subtopics and prerequisites."""
    prompt = PromptTemplate(
        input_variables=["topic", "selected_subtopics", "prereq_proficiency"],
        template=ROADMAP_TEMPLATE
    )
    response = llm.invoke(prompt.format(
        topic=topic,
        selected_subtopics=json.dumps(selected_subtopics),
        prereq_proficiency=json.dumps(prereq_proficiency)
    ))
    
    default_roadmap = {
        "sequence": ["Start with basics", "Practice exercises", "Advanced concepts"],
        "pitfalls": ["Take time to understand fundamentals"],
        "focus_areas": ["Core concepts"],
        "time_estimate": "2-4 weeks depending on dedication"
    }
    
    return parse_json_response(response.content, default_roadmap)

def display_roadmap(roadmap: Dict):
    """Display roadmap with consistent section headers."""
    sections = {
        "sequence": "Learning Sequence",
        "pitfalls": "Common Pitfalls",
        "focus_areas": "Key Focus Areas",
        "time_estimate": "Time Estimate"
    }
    
    for key, title in sections.items():
        if key in roadmap:
            st.subheader(title)
            content = roadmap[key]
            if isinstance(content, list):
                for item in content:
                    st.write(f"- {item}")
            else:
                st.write(content)

def main():
    st.title("AI E-Library Search Assistant")
    
    # Initialize session state
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
    if "llm" not in st.session_state:
        st.session_state.llm = initialize_llm()
    if "current_stage" not in st.session_state:
        st.session_state.current_stage = "topic_input"
    
    # Topic Input Stage
    if st.session_state.current_stage == "topic_input":
        st.header("What would you like to learn?")
        topic = st.text_input("Enter your subject or topic of interest:")
        if st.button("Next") and topic:
            with st.spinner("Generating prerequisites..."):
                st.session_state.topic = topic
                st.session_state.prerequisites = get_prerequisites(st.session_state.llm, topic)
                st.session_state.current_stage = "prerequisites"
                st.rerun()
    
    # Prerequisites Proficiency Stage
    elif st.session_state.current_stage == "prerequisites":
        st.header("Prerequisites Assessment")
        st.write("Please rate your proficiency in the following prerequisites:")
        prereq_proficiency = {}
        for prereq in st.session_state.prerequisites:
            level = st.select_slider(
                f"How well do you know {prereq}?",
                options=["Beginner", "Intermediate", "Advanced"],
                key=prereq
            )
            prereq_proficiency[prereq] = level
        
        if st.button("Generate Subtopics"):
            with st.spinner("Generating subtopics..."):
                st.session_state.prereq_proficiency = prereq_proficiency
                st.session_state.subtopics = get_subtopics(
                    st.session_state.llm,
                    st.session_state.topic,
                    prereq_proficiency
                )
                st.session_state.current_stage = "subtopics"
                st.rerun()
    
    # Subtopics Selection Stage
    elif st.session_state.current_stage == "subtopics":
        st.header("Select Subtopics")
        selected_subtopics = []
        for subtopic in st.session_state.subtopics:
            if st.checkbox(subtopic, key=subtopic):
                selected_subtopics.append(subtopic)
        
        if st.button("Generate Roadmap") and selected_subtopics:
            with st.spinner("Generating roadmap..."):
                st.session_state.selected_subtopics = selected_subtopics
                st.session_state.roadmap = generate_roadmap(
                    st.session_state.llm,
                    st.session_state.topic,
                    selected_subtopics,
                    st.session_state.prereq_proficiency
                )
                st.session_state.current_stage = "roadmap"
                st.rerun()
    
    # Roadmap and Chat Stage
    elif st.session_state.current_stage == "roadmap":
        st.header("Your Learning Roadmap")
        display_roadmap(st.session_state.roadmap)
        
        # Chat interface
        st.header("Have Questions? Ask Here!")
        user_input = st.text_input("Type your question:")
        if st.button("Ask") and user_input:
            with st.spinner("Generating response..."):
                prompt = PromptTemplate(
                    input_variables=["topic", "chat_history", "human_input", "prereq_proficiency", "selected_subtopics"],
                    template=CHATBOT_TEMPLATE
                )
                
                chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]
                
                response = st.session_state.llm.invoke(prompt.format(
                    topic=st.session_state.topic,
                    chat_history=chat_history,
                    human_input=user_input,
                    prereq_proficiency=json.dumps(st.session_state.prereq_proficiency),
                    selected_subtopics=json.dumps(st.session_state.selected_subtopics)
                ))
                
                st.session_state.memory.save_context(
                    {"human_input": user_input},
                    {"output": response.content}
                )
                
                # Display chat history
                for message in st.session_state.memory.load_memory_variables({})["chat_history"]:
                    if isinstance(message, HumanMessage):
                        st.write(f"You: {message.content}")
                    else:
                        st.write(f"Assistant: {message.content}")

if __name__ == "__main__":
    main()