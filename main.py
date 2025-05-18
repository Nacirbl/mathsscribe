import json
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

# --- Configuration ---
DEEPINFRA_API_KEY = "KOgI2z74uv6eO0gWPpnNQV7xOQvkaWDk"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai/"
LLM_MODEL_NAME = "google/gemma-3-27b-it"

# --- Initialize Flask App ---
app = Flask(__name__, static_folder="static", template_folder="templates")

# --- Initialize OpenAI Client ---
try:
    client = OpenAI(api_key=DEEPINFRA_API_KEY, base_url=DEEPINFRA_BASE_URL)
except Exception as e:
    print(f"[Error] Failed to initialize OpenAI client: {e}")
    client = None

# --- Homepage Route ---
@app.route("/")
def home():
    return render_template("index.html")

# --- Prompt Builder ---
def create_llm_prompt(user_description: str) -> list:
    return [
        {
            "role": "system",
            "content": (
                "You are a highly intelligent AI assistant specialized in mathematics and LaTeX.\n"
                "Your task is to:\n"
                "1. Convert the user's natural language description of a mathematical expression into a standard LaTeX formula.\n"
                "2. Provide a clear and concise explanation of this formula, its variables, and its common applications or significance.\n"
                "3. If the expression can be solved, simplified, evaluated, or if it's an equation, provide the solution, steps, or the result.\n"
                "Return your response strictly as a JSON object with the keys: latex_formula, explanation, solution.\n"
                "Do not include any markdown formatting or code fences like ```json."
            )
        },
        {"role": "user", "content": f'User\'s mathematical description: "{user_description}"'}
    ]

# --- API Endpoint ---
@app.route("/api/generate-formula", methods=["POST"])
def generate_formula():
    if not client:
        return jsonify({"error": "LLM client not initialized"}), 500

    try:
        data = request.get_json(force=True)
        description = data.get("description", "").strip()
        if not description:
            return jsonify({"error": "No description provided."}), 400

        messages = create_llm_prompt(description)
        print(f"[Request] Prompt: {description}")

        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        print(f"[LLM] Raw Output: {content}")

        try:
            parsed = json.loads(content)
            return jsonify({
                "latex_formula": parsed.get("latex_formula", ""),
                "explanation": parsed.get("explanation", ""),
                "solution": parsed.get("solution", "")
            })
        except json.JSONDecodeError:
            return jsonify({
                "latex_formula": "",
                "explanation": "",
                "solution": "",
                "error": "LLM returned non-JSON output. Response: " + content
            }), 500

    except Exception as e:
        print(f"[Error] {e}")
        return jsonify({"error": str(e)}), 500

# --- Run the app ---
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
