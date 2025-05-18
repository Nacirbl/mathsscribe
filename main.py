import json
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
import os

# --- Configuration ---
DEEPINFRA_API_KEY = "KOgI2z74uv6eO0gWPpnNQV7xOQvkaWDk"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai/"
LLM_MODEL_NAME = "google/gemma-3-27b-it"

# --- Flask App Initialization ---
app = Flask(__name__, static_folder="static", template_folder="templates")

# --- OpenAI Client Initialization ---
try:
    client = OpenAI(
        api_key=DEEPINFRA_API_KEY,
        base_url=DEEPINFRA_BASE_URL,
    )
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

# --- Home Route: Serve Frontend ---
@app.route("/")
def home():
    return render_template("index.html")

# --- Prompt Constructor ---
def create_llm_prompt(user_description: str) -> list:
    system_message = """
You are a highly intelligent AI assistant specialized in mathematics and LaTeX.
Your task is to:
1. Convert the user's natural language description of a mathematical expression into a standard LaTeX formula.
2. Provide a clear and concise explanation of this formula, its variables, and its common applications or significance.
3. If the expression can be solved, simplified, evaluated, or if it's an equation, provide the solution, steps, or the result. If it's a definition, theorem, or concept, explain its implications or provide a demonstrative example.
Return your response strictly as a JSON object with the following keys: "latex_formula", "explanation", "solution".
Do not include any markdown formatting (like ```json) around the JSON object itself.
"""
    user_message = f'User\'s mathematical description: "{user_description}"'
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

# --- API Endpoint ---
@app.route("/api/generate-formula", methods=["POST"])
def generate_formula():
    if not client:
        return jsonify({"error": "OpenAI client not initialized"}), 500

    data = request.get_json()
    description = data.get("description", "").strip()

    if not description:
        return jsonify({"error": "No description provided"}), 400

    messages = create_llm_prompt(description)

    try:
        print(f"Processing: {description}")
        chat_completion = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        response_content = chat_completion.choices[0].message.content
        print(f"LLM Response: {response_content}")

        try:
            parsed = json.loads(response_content)
            return jsonify({
                "latex_formula": parsed.get("latex_formula", "No LaTeX formula provided."),
                "explanation": parsed.get("explanation", "No explanation provided."),
                "solution": parsed.get("solution", "No solution provided.")
            })

        except json.JSONDecodeError:
            return jsonify({
                "latex_formula": "",
                "explanation": "",
                "solution": "",
                "error": f"LLM returned invalid JSON: {response_content}"
            })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

# --- Run Locally ---
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
