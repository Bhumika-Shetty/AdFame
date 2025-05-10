from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__)

# Configuration for the FastAPI service URL
FASTAPI_URL = os.environ.get("FASTAPI_SERVER_URL", "http://localhost:8000")

@app.route("/")
def index_get():
    return render_template("index.html", video_path=None, error_message=None)

@app.route("/generate_video", methods=["POST"])
def generate_video_post():
    video_path = None
    error_message = None
    prompt = request.form.get("prompt")
    if not prompt:
        error_message = "Prompt cannot be empty."
        return render_template("index.html", video_path=None, error_message=error_message)
    
    try:
        response = requests.post(f"{FASTAPI_URL}/generate", json={"prompt": prompt}, timeout=300) # Increased timeout for potentially long video generation
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        video_path = data.get("video_path") # This should be like "saved_videos/my_video.mp4"
        if not video_path:
            error_message = "Failed to get video path from the API."
        # The video_path is now directly usable with url_for('static', filename=video_path)
        # because 'saved_videos' will be a subdirectory within Flask's static folder due to volume mount
    except requests.exceptions.Timeout:
        error_message = "The video generation request timed out. Please try again with a simpler prompt or check the backend service."
    except requests.exceptions.RequestException as e:
        error_message = f"Error connecting to the video generation service: {e}"
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
    
    return render_template("index.html", video_path=video_path, error_message=error_message)

# Combined route for GET and POST to simplify HTML form action
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return generate_video_post()
    return index_get()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

