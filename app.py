import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
import os
from typing import List, Dict
import json
from dotenv import load_dotenv

load_dotenv()

def initialize_llm():
    """Initialize the Groq LLM with specified parameters."""
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192",
        temperature=0.5,
        max_tokens=4096,
    )

def get_prerequisites_prompt(topic: str) -> str:
    return f"""You are a helpful educational assistant. For the topic '{topic}', list exactly 3-5 prerequisite topics.
    
    Rules:
    1. Response must be valid JSON
    2. Each prerequisite must have a topic name and difficulty level
    3. Difficulty must be one of: "Basic", "Intermediate", "Advanced"
    
    Return only the JSON in this exact format, nothing else:
    {{
        "prerequisites": [
            {{"topic": "example topic", "level": "Basic"}},
            {{"topic": "another topic", "level": "Intermediate"}}
        ]
    }}"""

def get_chatbot_prompt(topic: str, context: str) -> str:
    return f"""You are a helpful educational assistant specializing in {topic}. 
    Use this context about the user's learning journey: {context}
    
    Provide clear, concise answers focused on {topic} and related concepts in 150 words.
    If asked about topics outside your expertise, guide the conversation back to {topic}."""
    
def get_subtopics_prompt(topic: str) -> str:
    return f"""You are a helpful educational assistant. For the topic '{topic}', list exactly 5 key subtopics to study.
    
    Rules:
    1. Response must be valid JSON
    2. Each subtopic should be brief but descriptive
    
    Return only the JSON in this exact format, nothing else:
    {{
        "subtopics": [
            "subtopic 1",
            "subtopic 2",
            "subtopic 3",
            "subtopic 4",
            "subtopic 5"
        ]
    }}"""

def get_roadmap_prompt(topic: str, subtopics: List[str]) -> str:
    subtopics_str = ", ".join(subtopics)
    return f"""Create a detailed learning roadmap for mastering '{topic}' with focus on: {subtopics_str}

    Rules:
    1. Response must be valid JSON
    2. Break down into weeks (4-6 weeks total)
    3. Each week should have clear goals and activities
    4. Include estimated time commitment per week
    5. Include specific practice exercises and projects
    
    Return only the JSON in this exact format, nothing else:
    {{
        "roadmap": [
            {{
                "week": 1,
                "goals": ["goal1", "goal2"],
                "activities": ["activity1", "activity2"],
                "exercises": ["exercise1", "exercise2"],
                "project": "project description",
                "hours_per_week": 10
            }}
        ]
    }}"""

def get_content_prompt(subtopics: List[str]) -> str:
    subtopics_str = ", ".join(subtopics)
    return f"""Provide a comprehensive learning summary for these subtopics: {subtopics_str}

    For each subtopic:
    1. Start with a clear overview (2-3 sentences)
    2. List 3-4 key concepts with brief explanations
    3. Include 2-3 practical examples with code or detailed steps
    4. Suggest hands-on exercises or mini-projects
    5. Include common pitfalls and how to avoid them
    
    Structure the response with clear headings and bullet points.
    Focus on practical applications and real-world relevance.
    Include difficulty rating for each concept (Basic/Intermediate/Advanced)."""

def get_resources_prompt(topic: str, subtopics: List[str]) -> str:
    subtopics_str = ", ".join(subtopics)
    return f"""You are a helpful educational assistant. Provide learning resources for '{topic}' focusing on: {subtopics_str}

    Rules:
    1. Response must be valid JSON
    2. Include exactly 3 textbooks and 3 papers
    3. Include exactly 1 YouTube video (full URL) for '{topic}' focusing on: {subtopics_str}
    4. Include 2 online courses
    5. Include 2 interactive learning platforms
    
    Return only the JSON in this exact format, nothing else:
    {{
        "textbooks": [
            {{"title": "Book Title 1", "author": "Author Name", "link": "https://example.com/book1"}},
            {{"title": "Book Title 2", "author": "Author Name", "link": "https://example.com/book2"}},
            {{"title": "Book Title 3", "author": "Author Name", "link": "https://example.com/book3"}}
        ],
        "papers": [
            {{"title": "Paper Title 1", "authors": "Authors", "link": "https://example.com/paper1"}},
            {{"title": "Paper Title 2", "authors": "Authors", "link": "https://example.com/paper2"}},
            {{"title": "Paper Title 3", "authors": "Authors", "link": "https://example.com/paper3"}}
        ],
        "youtube": "https://youtube.com/watch?v=FULL_VIDEO_URL",
        "courses": [
            {{"title": "Course Title 1", "platform": "Platform Name", "link": "https://example.com/course1"}},
            {{"title": "Course Title 2", "platform": "Platform Name", "link": "https://example.com/course2"}}
        ],
        "interactive_platforms": [
            {{"name": "Platform Name 1", "description": "Description", "link": "https://example.com/platform1"}},
            {{"name": "Platform Name 2", "description": "Description", "link": "https://example.com/platform2"}}
        ]
    }}"""

def extract_json_from_text(text: str) -> str:
    """Extract JSON content from text using a simple but reliable method."""
    try:
        # Find the first '{' and last '}'
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != -1:
            return text[start:end]
        return "{}"
    except Exception:
        return "{}"

def parse_json_response(response: str) -> Dict:
    """Safely parse JSON from LLM response with better error handling."""
    try:
        cleaned_response = response.strip()
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        try:
            json_str = extract_json_from_text(cleaned_response)
            return json.loads(json_str)
        except json.JSONDecodeError:
            st.warning("Failed to parse response. Using simplified format.")
            if '"prerequisites"' in response:
                return {"prerequisites": [{"topic": "Basic concepts", "level": "Basic"}]}
            elif '"subtopics"' in response:
                return {"subtopics": ["Basic concepts"]}
            elif '"textbooks"' in response:
                return {
                    "textbooks": [{"title": "Resource not available", "author": "N/A", "link": "#"}],
                    "papers": [{"title": "Resource not available", "authors": "N/A", "link": "#"}],
                    "youtube": "https://youtube.com/watch?v=dQw4w9WgXcQ",
                    "courses": [{"title": "Resource not available", "platform": "N/A", "link": "#"}],
                    "interactive_platforms": [{"name": "Resource not available", "description": "N/A", "link": "#"}]
                }
            return {}
        
def display_chatbot(llm, topic: str, context: str):
    """Display and handle the chatbot interface."""
    st.subheader("Chat with Your AI Learning Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your topic..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate and display assistant response
        with st.chat_message("assistant"):
            full_prompt = get_chatbot_prompt(topic, context) + "\n\nUser: " + prompt
            response = llm.invoke(full_prompt)
            st.markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})

def main():
    st.markdown(
    """
    <h1 style="text-align: center;">InfiLib</h1>
    """,
    unsafe_allow_html=True,
    )
    st.markdown(
    """
    <h1 style="text-align: center;">AI Powered E-Library Assistant</h1>
    """,
    unsafe_allow_html=True,
    )
    
    # Initialize session state
    if 'responses' not in st.session_state:
        st.session_state.responses = {}
    if 'prereq_levels' not in st.session_state:
        st.session_state.prereq_levels = {}
    
    try:
        llm = initialize_llm()
    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        return
    
    # Topic Selection
    topic = st.text_input("What topic would you like to study?")
    
    if topic:
        try:
            # Prerequisites Check
            if 'prerequisites' not in st.session_state.responses:
                with st.spinner("Loading prerequisites..."):
                    prerequisites_response = llm.invoke(get_prerequisites_prompt(topic))
                    st.session_state.responses['prerequisites'] = parse_json_response(prerequisites_response.content)
            
            prereqs_data = st.session_state.responses['prerequisites']
            if prereqs_data.get("prerequisites"):
                st.subheader("Prerequisites Assessment")
                st.write("Please indicate your level for each prerequisite:")
                
                # Create columns for a more compact layout
                cols = st.columns(2)
                for idx, prereq in enumerate(prereqs_data["prerequisites"]):
                    with cols[idx % 2]:
                        # Add dropdown for each prerequisite
                        prereq_key = f"prereq_level_{prereq['topic']}"
                        level = st.selectbox(
                            f"{prereq['topic']} (Recommended: {prereq['level']})",
                            options=["Beginner", "Intermediate", "Advanced"],
                            key=prereq_key
                        )
                        st.session_state.prereq_levels[prereq['topic']] = level
            
            # Subtopics Selection
            if 'subtopics' not in st.session_state.responses:
                with st.spinner("Loading subtopics..."):
                    subtopics_response = llm.invoke(get_subtopics_prompt(topic))
                    st.session_state.responses['subtopics'] = parse_json_response(subtopics_response.content)
            
            subtopics_data = st.session_state.responses['subtopics']
            if subtopics_data.get("subtopics"):
                st.subheader("Select Subtopics to Study")
                selected_subtopics = []
                for subtopic in subtopics_data["subtopics"]:
                    if st.checkbox(subtopic, key=f"subtopic_{subtopic}"):
                        selected_subtopics.append(subtopic)
            
                if selected_subtopics:
                    # Generate Learning Roadmap
                    if 'roadmap' not in st.session_state.responses:
                        with st.spinner("Generating learning roadmap..."):
                            roadmap_response = llm.invoke(get_roadmap_prompt(topic, selected_subtopics))
                            st.session_state.responses['roadmap'] = parse_json_response(roadmap_response.content)
                    
                    # Display Roadmap
                    roadmap_data = st.session_state.responses.get('roadmap', {})
                    if roadmap_data.get("roadmap"):
                        st.subheader("Learning Roadmap")
                        for week in roadmap_data["roadmap"]:
                            with st.expander(f"Week {week['week']} (Time Commitment: {week['hours_per_week']} hours)"):
                                st.write("🎯 Goals:")
                                for goal in week['goals']:
                                    st.write(f"- {goal}")
                                st.write("📚 Activities:")
                                for activity in week['activities']:
                                    st.write(f"- {activity}")
                                st.write("✍️ Practice Exercises:")
                                for exercise in week['exercises']:
                                    st.write(f"- {exercise}")
                                st.write("🚀 Project:")
                                st.write(week['project'])
                    
                    # Content Summary
                    st.subheader("Content Summary")
                    with st.spinner("Generating content summary..."):
                        content_response = llm.invoke(get_content_prompt(selected_subtopics))
                        st.write(content_response.content)
                    
                    # Resources and YouTube Link
                    if 'resources' not in st.session_state.responses:
                        with st.spinner("Finding learning resources..."):
                            resources_response = llm.invoke(get_resources_prompt(topic, selected_subtopics))
                            st.session_state.responses['resources'] = parse_json_response(resources_response.content)
                    
                    resources_data = st.session_state.responses['resources']
                    if resources_data:
                        st.subheader("📚 Recommended Textbooks")
                        for book in resources_data.get("textbooks", []):
                            st.markdown(f"- [{book['title']}]({book['link']}) by {book['author']}")
                        
                        st.subheader("📄 Relevant Research Papers")
                        for paper in resources_data.get("papers", []):
                            st.markdown(f"- [{paper['title']}]({book['link']}) by {paper['authors']}")
                        
                        st.subheader("🎓 Online Courses")
                        for course in resources_data.get("courses", []):
                            st.markdown(f"- [{course['title']}]({course['link']}) on {course['platform']}")
                        
                        st.subheader("💻 Interactive Learning Platforms")
                        for platform in resources_data.get("interactive_platforms", []):
                            st.markdown(f"- [{platform['name']}]({platform['link']}) - {platform['description']}")
                        
                        # Display YouTube Video
                        if youtube_url := resources_data.get("youtube"):
                            st.subheader("🎥 Recommended Tutorial")
                            st.video(youtube_url)
                            
            if any(st.session_state.prereq_levels):  # Only show chatbot if prerequisites have been assessed
                st.divider()
                # Create context from user's prerequisites assessment
                context = "User's prerequisite levels:\n" + \
                         "\n".join([f"{topic}: {level}" for topic, level in st.session_state.prereq_levels.items()])
                display_chatbot(llm, topic, context)
                            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please try refreshing the page or selecting different options.")

if __name__ == "__main__":
    main()