from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__)

# Configuration for the FastAPI service URL
FASTAPI_URL = os.environ.get("FASTAPI_SERVER_URL", "http://localhost:8000")

@app.route('/', methods=['GET', 'POST'])
def index():
    video_path = None
    error_message = None
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        if not prompt:
            error_message = "Prompt cannot be empty."
        else:
            try:
                response = requests.post(f"{FASTAPI_URL}/generate", json={"prompt": prompt})
                response.raise_for_status()  # Raise an exception for HTTP errors
                data = response.json()
                video_path = data.get('video_path')
                if not video_path:
                    error_message = "Failed to get video path from the API."
            except requests.exceptions.RequestException as e:
                error_message = f"Error connecting to the video generation service: {e}"
            except Exception as e:
                error_message = f"An unexpected error occurred: {e}"
    return render_template('index.html', video_path=video_path, error_message=error_message)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

