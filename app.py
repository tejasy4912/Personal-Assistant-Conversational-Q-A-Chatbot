from flask import Flask, request, jsonify, render_template
import openai
import json
from dotenv import load_dotenv
import os

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
    skills = "\n".join(
        [f"{key.capitalize()}: {', '.join(value)}" for key, value in profile_data['skills']['domains'].items()]
    )
    experience = "\n".join([
        f"{exp['role']} at {exp['organization']} ({exp['duration']}): {'; '.join(exp['responsibilities'])}"
        for exp in profile_data['experience']
    ])
    projects = "\n".join([
        f"{proj['title']} ({proj['duration']}): {' '.join(proj['description'])}"
        for proj in profile_data['projects']
    ])
    return (
        f"Name: {profile_data['name']}\n"
        f"Summary: {profile_data['summary']}\n"
        f"Education:\n"
        f"  - Master's: {profile_data['education']['masters']['degree']} in {profile_data['education']['masters']['concentration']} from {profile_data['education']['masters']['institution']} (GPA: {profile_data['education']['masters']['gpa']})\n"
        f"  - Bachelor's: {profile_data['education']['bachelors']['degree']} in {profile_data['education']['bachelors']['concentration']} from {profile_data['education']['bachelors']['institution']} (GPA: {profile_data['education']['bachelors']['gpa']})\n"
        f"Skills:\n{skills}\n"
        f"Experience:\n{experience}\n"
        f"Projects:\n{projects}\n"
        f"Certifications: {', '.join(profile_data['certifications'])}\n"
    )

# Load the JSON data
try:
    profile_data = load_profile_data("Profile_data.json")
    profile_context = build_detailed_context(profile_data)
except Exception as e:
    print(f"Error loading profile data: {str(e)}")
    profile_context = ""

# Chatbot function
def chatbot(user_input, conversation_history):
    # Add user input to conversation
    conversation_history.append({"role": "user", "content": user_input})

    try:
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_history
        )
        reply = response["choices"][0]["message"]["content"]

        # Add assistant reply to conversation
        conversation_history.append({"role": "assistant", "content": reply})

        return reply
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Flask routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("user_input")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Maintain a stateless conversation history
    conversation_history = [
        {"role": "system", "content": f"You are Tejas Pawar's professional assistant. Use the following profile context:\n{profile_context}"}
    ]
    reply = chatbot(user_input, conversation_history)
    if "An error occurred" in reply:
        return jsonify({"error": reply}), 500
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)