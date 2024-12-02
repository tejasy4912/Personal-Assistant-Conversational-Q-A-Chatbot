from flask import Flask, request, jsonify, render_template
import openai
import json
from dotenv import load_dotenv
import os
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)

# Load JSON profile data
def load_profile_data(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        raise Exception(f"Profile data file '{file_path}' not found.")
    except json.JSONDecodeError:
        raise Exception(f"Profile data file '{file_path}' contains invalid JSON.")

# Build detailed context
def build_detailed_context(profile_data):
    # Extract contact information
    contact_info = (
        f"Address: {profile_data['contact']['address']}\n"
        f"Phone: {profile_data['contact']['phone']}\n"
        f"Email: {profile_data['contact']['email']}\n"
        f"Links:\n"
        + "\n".join([f"  - {link['name']}: {link['url']}" for link in profile_data['contact']['links']])
    )

    # Extract work authorization details
    work_auth = (
        f"Work Authorization Status: {profile_data['work_authorization']['status']}\n"
        f"Details: {profile_data['work_authorization']['details']}\n"
        f"Sponsorship Preference: {profile_data['work_authorization']['sponsorship_preference']}\n"
        f"Employment Type: {profile_data['work_authorization']['employment_type']}\n"
    )

    # Extract education details
    education = (
        f"  - Master's: {profile_data['education']['masters']['degree']} in {profile_data['education']['masters']['concentration']} "
        f"from {profile_data['education']['masters']['institution']} (GPA: {profile_data['education']['masters']['gpa']}, "
        f"Graduation Date: {profile_data['education']['masters']['graduation_date']})\n"
        f"    Coursework: {', '.join(profile_data['education']['masters']['coursework'])}\n"
        f"  - Bachelor's: {profile_data['education']['bachelors']['degree']} in {profile_data['education']['bachelors']['concentration']} "
        f"from {profile_data['education']['bachelors']['institution']} (GPA: {profile_data['education']['bachelors']['gpa']}, "
        f"Graduation Date: {profile_data['education']['bachelors']['graduation_date']})\n"
        f"    Coursework: {', '.join(profile_data['education']['bachelors']['coursework'])}\n"
    )

    # Extract experience details
    experience = "\n".join([
        f"{exp['role']} at {exp['organization']} ({exp['duration']}): {'; '.join(exp['responsibilities'])}"
        for exp in profile_data['experience']
    ])

    # Extract project details
    projects = "\n".join([
        f"{proj['title']} ({proj['duration']}): {' '.join(proj['description'])}"
        for proj in profile_data['projects']
    ])

    # Extract leadership details
    leadership = "\n".join([
        f"{lead['role']} at {lead['organization']} ({lead['duration']}): {'; '.join(lead['responsibilities'])}"
        for lead in profile_data['leadership']
    ])

    # Extract volunteer details
    volunteer = "\n".join([
        f"{vol['role']} at {vol['organization']} ({vol['duration']}): {'; '.join(vol['responsibilities'])}"
        for vol in profile_data['volunteer']
    ])

    # Extract certifications
    certifications = ", ".join(profile_data['certifications'])

    # Extract publications
    publications = "\n".join([
        f"{pub['title']} ({pub['conference']}): Published in {pub['publication']}, awarded {pub['award']}"
        for pub in profile_data['publications']
    ])

    # Extract skills
    skills = "\n".join(
        [f"{key.capitalize()}: {', '.join(value)}" for key, value in profile_data['skills']['domains'].items()]
    )

    # Build the detailed profile context
    return (
        f"Name: {profile_data['name']}\n"
        f"Contact:\n{contact_info}\n"
        f"Summary: {profile_data['summary']}\n"
        f"Work Authorization:\n{work_auth}\n"
        f"Education:\n{education}\n"
        f"Skills:\n{skills}\n"
        f"Experience:\n{experience}\n"
        f"Projects:\n{projects}\n"
        f"Leadership:\n{leadership}\n"
        f"Volunteer Work:\n{volunteer}\n"
        f"Certifications: {certifications}\n"
        f"Publications:\n{publications}\n"
    )


# Load the JSON data
try:
    profile_data = load_profile_data("Profile_data.json")
    profile_context = build_detailed_context(profile_data)
except Exception as e:
    print(f"Error loading profile data: {str(e)}")
    profile_context = ""

# Initialize LangChain components
memory = ConversationBufferMemory(return_messages=True)
llm = ChatOpenAI(model="gpt-4", temperature=0)
conversation = ConversationChain(llm=llm, memory=memory)

# Flask routes
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("user_input")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    try:
        # Add profile context at the beginning of the conversation if it's the first query
        if len(memory.chat_memory.messages) == 0:
            intro_context = (
                "You are Sarah, Tejas Pawar's dedicated professional assistant. "
                "Your primary role is to use the provided profile context to answer queries with clarity, precision, and relevance. keep your response concise and to the point "
                "Ensure that your responses are concise, structured, and tailored to the question. Focus on presenting Tejas's experience, skills, projects, "
                "and achievements in a way that emphasizes his strengths and aligns with the query. "
                "If a job description is provided, analyze and match it with Tejas's profile to highlight key strengths, relevant skills, and accomplishments "
                "that make him a strong candidate. Always provide thoughtful and well-supported responses by leveraging the available information. "
                "When addressing topics not explicitly detailed in the profile, focus on related skills, coursework, or transferable expertise to provide a positive and comprehensive response. "
                "Redirect specific or detailed inquiries gracefully by encouraging further engagement, maintaining a professional and supportive tone. Never say that details are not availabe in th eprofessional profile instead ask them to reach out Tejas directly and their provide my email and meeting link"
                "Stay strictly within the context of Tejas's professional assistant role and avoid addressing unrelated or out-of-context questions.\n\n"
                f"{profile_context}"
            )

            memory.chat_memory.add_user_message(intro_context)

        # Handle user queries dynamically
        if user_input.lower() in ["hi", "hello", "who are you"]:
            reply = "Hi, I’m Sarah. Let me know how I can assist you with Tejas's profile, experience, or projects."
        else:
            # Generate a response using LangChain
            reply = conversation.run(user_input)

        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use the PORT provided by Render or default to 5000
    app.run(host="0.0.0.0", port=port, debug=False)
