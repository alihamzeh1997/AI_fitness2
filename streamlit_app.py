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
import time
from typing import List, Tuple, Optional, Dict, Any

# Configure page
st.set_page_config(
    page_title="Fitness Video Analyzer",
    page_icon="💪",
    layout="wide"
)

# Set up OpenRouter API
def setup_openrouter_api() -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    """Set up the OpenRouter API connection with proper error handling."""
    api_key = os.getenv('OPENROUTER_API_KEY') or st.secrets.get("OPENROUTER_API_KEY", None)
    
    if not api_key:
        st.error("OPENROUTER_API_KEY not found. Please set it in environment variables or Streamlit secrets.")
        return None, None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv('SITE_URL', 'https://fitness-analyzer.example.com'),  # Required by OpenRouter
        "X-Title": "Fitness Video Analyzer"  # Helps with API usage tracking
    }
    base_url = "https://openrouter.ai/api/v1"
    
    return base_url, headers

# Smart resize function
def smart_resize(frame: np.ndarray, min_dim: int = 300, max_dim: int = 400) -> np.ndarray:
    """Resize the frame while maintaining aspect ratio within constraints."""
    h, w = frame.shape[:2]
    longer = max(h, w)
    scale = min(max_dim, max(min_dim, longer)) / longer
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)  # Added better interpolation

# Draw timestamp on frame
def draw_timestamp(frame: np.ndarray, timestamp: float) -> np.ndarray:
    """Add timestamp text to a frame."""
    # Create a copy to avoid modifying the original
    result = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Time: {timestamp:.2f}s"
    
    # Draw shadow/outline for better visibility on any background
    cv2.putText(result, text, (6, 21), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(result, text, (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    return result

# Extract frames from video with frame limit and adjustable FPS
def extract_frames(
    video_path: str, 
    fps: int = 4, 
    min_dim: int = 300, 
    max_dim: int = 400, 
    max_frames: int = 30
) -> List[Tuple[float, np.ndarray]]:
    """Extract frames from a video at specified intervals with timestamps."""
    try:
        vr = VideoReader(video_path)
        video_fps = vr.get_avg_fps()
        num_frames = len(vr)
        
        if num_frames == 0:
            st.error("Video contains no frames.")
            return []
            
        duration = num_frames / video_fps
        
        frames_with_timestamps = []
        
        # For very short videos, increase sampling rate
        if duration < 1.0 and num_frames > 1:
            # Take frames evenly across the video
            indices = np.linspace(0, num_frames-1, min(max_frames, num_frames)).astype(int)
            for idx in indices:
                timestamp = idx / video_fps
                frame = vr[idx].asnumpy()
                frame = smart_resize(frame, min_dim, max_dim)
                frame = draw_timestamp(frame, timestamp)
                frames_with_timestamps.append((timestamp, frame))
            return frames_with_timestamps
        
        # Regular interval sampling for normal-length videos
        frame_interval = video_fps / fps
        total_frames_to_extract = min(max_frames, int(duration * fps))
        
        # Distribute frames evenly across video duration
        for i in range(total_frames_to_extract):
            # Calculate frame index for even distribution
            progress = i / (total_frames_to_extract - 1 if total_frames_to_extract > 1 else 1)
            frame_index = min(int(progress * (num_frames - 1)), num_frames - 1)
            timestamp = frame_index / video_fps
            
            frame = vr[frame_index].asnumpy()
            frame = smart_resize(frame, min_dim, max_dim)
            frame = draw_timestamp(frame, timestamp)
            frames_with_timestamps.append((timestamp, frame))
        
        return frames_with_timestamps
    except Exception as e:
        st.error(f"Error extracting frames: {str(e)}")
        return []

# Convert image to base64 and calculate size
def image_to_base64(image: np.ndarray, calculate_size: bool = False) -> Tuple[str, Optional[float]]:
    """Convert an image to base64 and optionally calculate its size."""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])  # Added quality parameter
    
    if calculate_size:
        size_kb = len(buffer) / 1024
        return base64.b64encode(buffer).decode('utf-8'), size_kb
    
    return base64.b64encode(buffer).decode('utf-8'), None

# Analyze frames with OpenRouter
def analyze_with_openrouter_individual_frames(
    base_url: Optional[str], 
    headers: Optional[Dict[str, str]], 
    video_file,
    frames_per_second: int = 4, 
    max_frames: int = 30, 
    model_name: str = "google/gemini-2.5-pro-exp-03-25"
) -> Tuple[str, List[Dict[str, Any]]]:
    """Analyze video frames using OpenRouter API."""
    if not base_url or not headers:
        return "Error: API setup failed.", []
    
    start_time = time.time()
    
    # Create a temporary file from the uploaded file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        tmpfile.write(video_file.read())
        tmpfile_path = tmpfile.name
    
    try:
        # Check if file is valid video
        try:
            vr = VideoReader(tmpfile_path)
            video_fps = vr.get_avg_fps()
            num_frames = len(vr)
            total_duration_sec = round(num_frames / video_fps)
        except Exception as e:
            os.unlink(tmpfile_path)
            return f"Error loading video: {str(e)}. Please check the file format.", []
        
        frames = extract_frames(tmpfile_path, fps=frames_per_second, max_frames=max_frames)
        
        if not frames:
            os.unlink(tmpfile_path)
            return "Error: No frames could be extracted from the video.", []
        
        Role = (
            "You're a fitness expert analyzing fitness videos. "
            "Focus on user movement across timestamped frames extracted from the video."
        )
        
        TaskPrompt = (
            f"Analyze a sequence of {len(frames)} frames from a fitness video (original duration: ~{total_duration_sec}s). "
            f"Showing {frames_per_second} frames/second, each with a timestamp.\n\n"
            "Steps:\n"
            "1. **Identify Exercise**: Determine the exercise (e.g., squats, push-ups) using full-body positioning and movement patterns over time.\n"
            "2. **Count Reps**: Track repetitions by analyzing joint movement and body posture across timestamps. Confirm rep completion.\n"
            "3. **Assess Tempo**: Calculate rep duration and categorize as slow, moderate, or fast. Note tempo changes.\n"
            "4. **Evaluate Form**: Assess posture, alignment, range of motion, and flag issues.\n\n"
            "Refer to timestamps, use biomechanical terms, and avoid assumptions. "
            f"Note: Only {len(frames)} frames are provided at {frames_per_second} FPS, so analysis may be limited if the video is longer."
        )
        
        OutputFormat = (
            "Respond in this format:\n"
            "- Exercise identified:\n"
            "- Total repetition count (with timestamps):\n"
            "- Tempo assessment:\n"
            "- Form evaluation:\n"
            "- Recommendations:\n"
            "- Reasoning:"
        )
        
        Content = [
            {"role": "system", "content": Role + "\n" + TaskPrompt + "\n" + OutputFormat}
        ]
        
        # Calculate duration covered by frames
        num_frames = len(frames)
        duration_covered = num_frames / frames_per_second if num_frames > 0 else 0
        
        # Display frame info
        st.info(f"Extracted {num_frames} frames covering {duration_covered:.2f} seconds of video")
        
        # Create a visual preview of frames
        if frames:
            cols = st.columns(min(5, len(frames)))
            for i, ((timestamp, frame), col) in enumerate(zip(frames[:5], cols)):
                col.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    caption=f"Frame {i+1}: {timestamp:.2f}s",
                    use_column_width=True
                )
            
            # Show first frame details
            timestamp, frame = frames[0]
            h, w = frame.shape[:2]
            frame_base64, size_kb = image_to_base64(frame, calculate_size=True)
            st.write(f"**Frame dimensions:** {w}x{h} pixels, ~{size_kb:.2f} kB per frame")
        
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
        
        # Show progress bar for API request
        with st.spinner(f"Analyzing with {model_name.split('/')[-1]}..."):
            try:
                response = requests.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": model_name,
                        "messages": Content,
                        "temperature": 0.1,
                        "top_p": 0.01,
                        "max_tokens": 1024  # Add token limit
                    },
                    timeout=120  # Add timeout
                )
                response.raise_for_status()
                
                processing_time = time.time() - start_time
                st.success(f"Analysis completed in {processing_time:.2f} seconds")
                
                return response.json()['choices'][0]['message']['content'], Content
            except requests.exceptions.Timeout:
                return "Error: Request timed out. The video might be too complex or the service is busy.", Content
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 400:
                    try:
                        error_msg = e.response.json().get("error", {}).get("message", str(e))
                        return f"API Error: {error_msg}", Content
                    except:
                        return f"API Error: {str(e)}", Content
                elif e.response.status_code == 401:
                    return "Error: Authentication failed. Check your API key.", Content
                elif e.response.status_code == 429:
                    return "Error: Rate limit exceeded. Try again later.", Content
                else:
                    return f"HTTP Error: {str(e)}", Content
            except Exception as e:
                return f"Error analyzing with OpenRouter: {str(e)}", Content
    finally:
        # Clean up temp file
        if os.path.exists(tmpfile_path):
            os.unlink(tmpfile_path)

def display_sidebar():
    """Configure and display sidebar content."""
    with st.sidebar:
        st.title("ℹ️ About")
        st.markdown("""
        **Fitness Video Analyzer** helps you:
        - Identify exercises from video
        - Count repetitions
        - Analyze movement tempo
        - Evaluate exercise form
        
        Powered by OpenRouter AI models.
        """)
        
        st.markdown("---")
        
        # API key input in sidebar with proper security
        api_key = st.text_input(
            "OpenRouter API Key", 
            type="password",
            help="Get your API key from https://openrouter.ai",
            key="api_key_input"
        )
        
        if api_key:
            # Securely store API key in session state
            if "OPENROUTER_API_KEY" not in st.session_state or st.session_state.OPENROUTER_API_KEY != api_key:
                st.session_state.OPENROUTER_API_KEY = api_key
                os.environ["OPENROUTER_API_KEY"] = api_key
                st.success("API key set!")
        
        st.markdown("---")
        st.markdown("Created by [Your Name]. v1.0.0")

# Streamlit app
def main():
    display_sidebar()
    
    st.title("💪 Fitness Video Analyzer")
    st.write("Upload a fitness video to analyze exercises, count reps, and evaluate form using AI.")
    
    # Define model options, grouped by provider and free/paid status
    model_options = {
        "Free Models": {
            "Google": [
                ("google/gemini-2.5-pro-exp-03-25:free", "Gemini 2.5 Pro (Recommended)"),
                ("google/gemini-2.0-flash-thinking-exp:free", "Gemini 2.0 Flash"),
                ("google/gemma-3-27b-it:free", "Gemma 3 27B")
            ],
            "Meta AI": [
                ("meta-llama/llama-3.2-11b-vision-instruct:free", "Llama 3.2 11B Vision"),
                ("meta-llama/llama-4-maverick:free", "Llama 4 Maverick")
            ],
            "Other": [
                ("allenai/molmo-7b-d:free", "AllenAI Molmo 7B"),
                ("bytedance-research/ui-tars-72b:free", "ByteDance TARS 72B"),
                ("mistralai/mistral-small-3.1-24b-instruct:free", "Mistral Small 3.1 24B"),
                ("moonshotai/kimi-vl-a3b-thinking:free", "Moonshot Kimi VL"),
                ("qwen/qwen2.5-vl-32b-instruct:free", "Qwen 2.5 VL 32B")
            ]
        },
        "Premium Models": {
            "Anthropic": [
                ("anthropic/claude-3.7-sonnet:thinking", "Claude 3.7 Sonnet Thinking"),
                ("anthropic/claude-3.7-sonnet", "Claude 3.7 Sonnet"),
                ("anthropic/claude-3.5-haiku:beta", "Claude 3.5 Haiku"),
                ("anthropic/claude-3-opus", "Claude 3 Opus")
            ],
            "OpenAI": [
                ("openai/chatgpt-4o-latest", "GPT-4o Latest"),
                ("openai/gpt-4.1", "GPT-4.1"),
                ("openai/gpt-4o-mini-2024-07-18", "GPT-4o Mini")
            ],
            "Other": [
                ("microsoft/phi-4-multimodal-instruct", "Microsoft Phi-4 Multimodal"),
                ("mistralai/pixtral-large-2411", "Mistral Pixtral Large"),
                ("x-ai/grok-2-vision-1212", "xAI Grok-2 Vision"),
                ("x-ai/grok-vision-beta", "xAI Grok Vision Beta")
            ]
        }
    }
    
    # Create tabs for file upload and settings
    tab1, tab2 = st.tabs(["Upload Video", "Advanced Settings"])
    
    with tab1:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # File uploader with better guidance
            uploaded_file = st.file_uploader(
                "Upload a fitness video",
                type=["mp4", "mov", "avi", "mkv"],
                help="For best results, use a clear video showing the full body movement",
                key="video_uploader"
            )
            
            # Sample video option
            use_sample = st.checkbox("Use sample video instead")
            
            if use_sample:
                # This would require a sample video to be included with your app
                # For now, just show a message
                st.info("Sample video feature will be available soon")
        
        with col2:
            # Model selection - simplified UI
            model_category = st.radio(
                "Model Type",
                options=["Free Models", "Premium Models"],
                horizontal=True,
                help="Free models are available to all users. Premium models require OpenRouter credits."
            )
            
            # Generate model options based on category
            model_providers = list(model_options[model_category].keys())
            provider = st.selectbox("Provider", options=model_providers)
            
            # Get display names and model IDs for the selected provider
            provider_models = model_options[model_category][provider]
            model_display_names = [model[1] for model in provider_models]
            model_ids = [model[0] for model in provider_models]
            
            # Select model by display name
            selected_display_name = st.selectbox(
                "Model",
                options=model_display_names,
                index=0 if model_category == "Free Models" and provider == "Google" else 0
            )
            
            # Map display name back to model ID
            selected_index = model_display_names.index(selected_display_name)
            model_name = model_ids[selected_index]
    
    with tab2:
        # Advanced settings
        col1, col2 = st.columns(2)
        
        with col1:
            frames_per_second = st.slider(
                "Frames per second", 
                min_value=1, 
                max_value=10, 
                value=4,
                help="Higher values give more detailed analysis but use more API credits"
            )
            
            min_dimension = st.slider(
                "Minimum frame dimension (px)",
                min_value=200,
                max_value=500,
                value=300,
                help="Smaller values reduce API usage but may affect analysis quality"
            )
            
        with col2:
            max_frames = st.slider(
                "Maximum frames to analyze", 
                min_value=5, 
                max_value=60, 
                value=20,
                help="More frames improve analysis but increase API usage and processing time"
            )
            
            max_dimension = st.slider(
                "Maximum frame dimension (px)",
                min_value=300,
                max_value=800,
                value=400,
                help="Larger values may improve analysis but increase API usage"
            )
    
    # Analysis button
    analyze_button = st.button(
        "🔍 Analyze Video",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_file and not use_sample
    )
    
    # Create container for results
    results_container = st.container()
    
    if analyze_button:
        if uploaded_file:
            with results_container:
                # Check API key
                if not os.getenv('OPENROUTER_API_KEY') and not st.session_state.get('OPENROUTER_API_KEY'):
                    st.error("⚠️ Please enter your OpenRouter API key in the sidebar.")
                    return
                
                with st.spinner("Processing video..."):
                    base_url, headers = setup_openrouter_api()
                    
                    if base_url and headers:
                        # Create tabs for analysis and debug info
                        result_tab1, result_tab2 = st.tabs(["Analysis Results", "Debug Info"])
                        
                        with result_tab1:
                            response_text, content = analyze_with_openrouter_individual_frames(
                                base_url, headers, uploaded_file,
                                frames_per_second=frames_per_second,
                                max_frames=max_frames,
                                model_name=model_name
                            )
                            
                            if response_text.startswith("Error"):
                                st.error(response_text)
                            else:
                                st.markdown("### Analysis Results")
                                st.markdown(response_text)
                                
                                # Add export options
                                export_col1, export_col2 = st.columns(2)
                                with export_col1:
                                    st.download_button(
                                        "📄 Download as Text",
                                        response_text,
                                        file_name="fitness_analysis.txt",
                                        mime="text/plain"
                                    )
                                with export_col2:
                                    # Export as JSON with metadata
                                    export_data = {
                                        "analysis": response_text,
                                        "metadata": {
                                            "model": model_name,
                                            "frames_analyzed": len(content) - 1 if content else 0,
                                            "frames_per_second": frames_per_second,
                                            "date": time.strftime("%Y-%m-%d %H:%M:%S")
                                        }
                                    }
                                    st.download_button(
                                        "🔄 Download as JSON",
                                        json.dumps(export_data, indent=2),
                                        file_name="fitness_analysis.json",
                                        mime="application/json"
                                    )
                        
                        with result_tab2:
                            st.write(f"**Model used:** {model_name}")
                            st.write(f"**Frames analyzed:** {len(content) - 1 if content else 0}")
                            st.write(f"**Analysis timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    else:
                        st.error("Cannot proceed without a valid API key.")
                        
        elif use_sample:
            st.info("Sample video feature will be available soon.")
        else:
            st.warning("Please upload a video file first.")

if __name__ == "__main__":
    # Initialize session state
    if "OPENROUTER_API_KEY" not in st.session_state:
        st.session_state.OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
    
    main()