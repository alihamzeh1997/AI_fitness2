import streamlit as st
import os
import cv2
import numpy as np
import requests
from PIL import Image
import tempfile
from decord import VideoReader
import base64
import json

# Set up OpenRouter API
def setup_openrouter_api():
    api_key = os.getenv('OPENROUTER_API_KEY') or st.secrets.get("OPENROUTER_API_KEY", None)
    if not api_key:
        st.error("OPENROUTER_API_KEY not found. Please set it in environment variables or Streamlit secrets.")
        return None, None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    base_url = "https://openrouter.ai/api/v1"
    
    return base_url, headers

# Smart resize function
def smart_resize(frame, min_dim=300, max_dim=400):
    h, w = frame.shape[:2]
    longer = max(h, w)
    scale = min(max_dim, max(min_dim, longer)) / longer
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(frame, new_size)

# Draw timestamp on frame
def draw_timestamp(frame, timestamp):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Time: {timestamp:.2f}s"
    cv2.putText(frame, text, (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame

# Extract frames from video with frame limit and 4 FPS
def extract_frames(video_path, fps=4, min_dim=300, max_dim=400, max_frames=30):
    vr = VideoReader(video_path)
    video_fps = vr.get_avg_fps()
    num_frames = len(vr)
    duration = num_frames / video_fps
    total_seconds = int(duration)
    
    frames_with_timestamps = []
    frame_count = 0
    
    for sec in range(total_seconds):
        for i in range(fps):
            if frame_count >= max_frames:
                return frames_with_timestamps
            timestamp = sec + i / fps
            frame_index = int(timestamp * video_fps)
            if frame_index < num_frames:
                frame = vr[frame_index].asnumpy()
                frame = smart_resize(frame, min_dim, max_dim)
                frame = draw_timestamp(frame, timestamp)
                frames_with_timestamps.append((timestamp, frame))
                frame_count += 1
    
    return frames_with_timestamps

# Convert image to base64 and calculate size
def image_to_base64(image, calculate_size=False):
    _, buffer = cv2.imencode('.jpg', image)
    if calculate_size:
        size_kb = len(buffer) / 1024
        return base64.b64encode(buffer).decode('utf-8'), size_kb
    return base64.b64encode(buffer).decode('utf-8'), None

# Analyze frames with OpenRouter
def analyze_with_openrouter_individual_frames(base_url, headers, video_path, frames_per_second=4, max_frames=30, model_name="google/gemini-2.5-pro-exp-03-25"):
    if not base_url or not headers:
        return "Error: API setup failed.", []
    
    Role = (
        "You're a fitness expert analyzing fitness videos. "
        "Focus on user movement across timestamped frames extracted from the video."
    )
    
    vr = VideoReader(video_path)
    video_fps = vr.get_avg_fps()
    num_frames = len(vr)
    total_duration_sec = round(num_frames / video_fps)
    
    TaskPrompt = (
        f"Analyze a sequence of up to {max_frames} frames from a fitness video (duration: ~{total_duration_sec}s). "
        f"Showing {frames_per_second} frames/second, each with a timestamp.\n\n"
        "Steps:\n"
        "1. **Identify Exercise**: Determine the exercise (e.g., squats, push-ups) using full-body positioning and movement patterns over time.\n"
        "2. **Count Reps**: Track repetitions by analyzing joint movement and body posture across timestamps. Confirm rep completion.\n"
        "3. **Assess Tempo**: Calculate rep duration and categorize as slow, moderate, or fast. Note tempo changes.\n"
        "4. **Evaluate Form**: Assess posture, alignment, range of motion, and flag issues.\n\n"
        "Refer to timestamps, use biomechanical terms, and avoid assumptions. "
        f"Note: Only {max_frames} frames are provided at {frames_per_second} FPS, so analysis may be limited if the video is longer."
    )
    
    OutputFormat = (
        "Respond in this format:\n"
        "- Exercise identified:\n"
        "- Total repetition count (with timestamps):\n"
        "- Tempo assessment:\n"
        "- Form evaluation:\n"
        "- Reasoning:"
    )
    
    Content = [
        {"role": "system", "content": Role + "\n" + TaskPrompt + "\n" + OutputFormat}
    ]
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        tmpfile.write(video_path.read())
        tmpfile_path = tmpfile.name
    
    frames = extract_frames(tmpfile_path, fps=frames_per_second, max_frames=max_frames)
    
    # Calculate duration covered by frames
    num_frames = len(frames)
    duration_covered = num_frames / frames_per_second if num_frames > 0 else 0
    
    # Display frame info
    st.write(f"**Feeding {num_frames} frames to the model, covering {duration_covered:.2f} seconds, using model: {model_name}**")
    
    # Display first frame size
    if frames:
        timestamp, frame = frames[0]
        h, w = frame.shape[:2]
        frame_base64, size_kb = image_to_base64(frame, calculate_size=True)
        st.write(f"**First frame (Timestamp {timestamp:.2f}s):** {h}x{w} pixels, {size_kb:.2f} kB")
    else:
        st.write("**No frames extracted.**")
    
    # Build content for API
    for i, (timestamp, frame) in enumerate(frames):
        frame_base64, _ = image_to_base64(frame, calculate_size=False)
        Content.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Frame {i+1}: Timestamp {timestamp:.2f}s. Analyze in sequence context."
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{frame_base64}"
                }
            ]
        })
    
    # Clean up temp file
    os.unlink(tmpfile_path)
    
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={
                "model": model_name,
                "messages": Content,
                "temperature": 0.1,
                "top_p": 0.01
            }
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'], Content
    except Exception as e:
        return f"Error analyzing with OpenRouter: {str(e)}", Content

# Streamlit app
def main():
    st.title("Fitness Video Analyzer")
    st.write("Upload a video to analyze exercises using OpenRouter's AI models.")
    
    # Define model options, sorted by provider
    model_options = {
        "Free Models": {
            "AllenAI": ["allenai/molmo-7b-d:free"],
            "ByteDance": ["bytedance-research/ui-tars-72b:free"],
            "Google": [
                "google/gemini-2.0-flash-thinking-exp:free",
                "google/gemini-2.5-pro-exp-03-25:free",
                "google/gemma-3-27b-it:free"
            ],
            "Meta AI": [
                "meta-llama/llama-3.2-11b-vision-instruct:free",
                "meta-llama/llama-4-maverick:free"
            ],
            "Mistral AI": ["mistralai/mistral-small-3.1-24b-instruct:free"],
            "Moonshot AI": ["moonshotai/kimi-vl-a3b-thinking:free"],
            "Qwen": ["qwen/qwen2.5-vl-32b-instruct:free"]
        },
        "Not Free Models": {
            "Anthropic": [
                "anthropic/claude-3-opus",
                "anthropic/claude-3.5-haiku:beta",
                "anthropic/claude-3.7-sonnet",
                "anthropic/claude-3.7-sonnet:thinking"
            ],
            "Microsoft": ["microsoft/phi-4-multimodal-instruct"],
            "Mistral AI": ["mistralai/pixtral-large-2411"],
            "OpenAI": [
                "openai/chatgpt-4o-latest",
                "openai/gpt-4.1",
                "openai/gpt-4o-mini-2024-07-18"
            ],
            "xAI": [
                "x-ai/grok-2-vision-1212",
                "x-ai/grok-vision-beta"
            ]
        }
    }
    
    # Create dropdown options
    dropdown_options = []
    for category in ["Free Models", "Not Free Models"]:
        dropdown_options.append(f"--- {category} ---")
        providers = sorted(model_options[category].keys())  # Sort providers alphabetically
        for provider in providers:
            for model in sorted(model_options[category][provider]):  # Sort models within provider
                dropdown_options.append(f"{provider}: {model}")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    
    # Input fields
    max_frames = st.number_input("Maximum number of frames", min_value=1, max_value=100, value=20)
    model_selection = st.selectbox(
        "Select model",
        options=dropdown_options,
        index=dropdown_options.index("Google: google/gemini-2.5-pro-exp-03-25:free"),
        help="Choose a vision-capable model from OpenRouter."
    )
    
    # Extract model name (remove provider prefix)
    model_name = model_selection.split(": ", 1)[1] if ": " in model_selection else model_selection
    
    # Analyze button
    if st.button("Analyze Video") and uploaded_file:
        with st.spinner("Processing video..."):
            base_url, headers = setup_openrouter_api()
            if base_url and headers:
                response_text, _ = analyze_with_openrouter_individual_frames(
                    base_url, headers, uploaded_file,
                    frames_per_second=4, max_frames=max_frames, model_name=model_name
                )
                st.markdown("### Analysis Results")
                st.markdown(response_text)
            else:
                st.error("Cannot proceed without a valid API key.")
    elif not uploaded_file and st.button("Analyze Video"):
        st.warning("Please upload a video file first.")

if __name__ == "__main__":
    main()