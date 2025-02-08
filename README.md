# InfiLib: AI-Powered E-Library Assistant  

## Overview  

InfiLib is an AI-powered e-learning assistant designed to provide structured learning paths, prerequisite assessments, study roadmaps, content summaries, and resource recommendations for various topics. It leverages large language models to generate relevant learning materials and facilitate interactive learning experiences.  

Checkout the live version !
- [Our Landing Page](https://sweet-cobbler-380a8b.netlify.app/) or
- [Application link](https://rakheshkrishna2005-infilib.hf.space/)

## Features  

- **Prerequisite Assessment**: Identifies essential concepts required to understand a topic and allows users to self-evaluate their proficiency.  
- **Subtopic Selection**: Breaks down a subject into key subtopics, enabling users to focus on specific areas of interest.  
- **Learning Roadmap**: Generates a structured study plan with weekly goals, activities, and practice exercises.  
- **Content Summaries**: Provides concise explanations, key concepts, examples, and potential pitfalls for selected subtopics.  
- **Resource Recommendations**: Suggests textbooks, research papers, online courses, and interactive learning platforms.  
- **AI-Powered Chat Assistant**: Enables users to engage in topic-specific discussions with an AI assistant.  

## Installation  

### Prerequisites  
Ensure the following dependencies are installed:  

- Python 3.8+  
- pip  
- Virtual environment (optional but recommended)  

### Setup Instructions  

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-repository/infilib.git
   cd infilib
   ```  

2. **Create and Activate a Virtual Environment** (Optional but recommended)  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```  

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```  

4. **Set Up Environment Variables**  
   Create a `.env` file in the project root and add the following:  
   ```ini
   GROQ_API_KEY=your_api_key_here
   ```  

5. **Run the Application**  
   ```bash
   streamlit run app.py
   ```  

## Usage  

1. Enter the topic you want to study.  
2. Review and assess prerequisite knowledge.  
3. Select subtopics for detailed study.  
4. View the generated learning roadmap.  
5. Read the content summaries for each subtopic.  
6. Explore recommended learning resources.  
7. Interact with the AI assistant for topic-related queries.  

## Technologies Used  

- **Streamlit** – Frontend interface for interactive user experience.  
- **LangChain** – Framework for managing language model interactions.  
- **Groq LLM** – Large language model for generating learning materials.  
- **Python** – Core programming language for application development.  

## Contributing  

Contributions are welcome. If you would like to improve the project, follow these steps:  

1. Fork the repository.  
2. Create a new branch (`feature-branch`).  
3. Make the necessary changes and commit them.  
4. Push the changes to your fork and create a pull request.  

## Contact  

For any inquiries or issues, please reach out via issues.  
