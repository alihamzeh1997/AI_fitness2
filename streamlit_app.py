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

# Initialize session state for API key
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv('OPENROUTER_API_KEY', '')

# Configure page
st.set_page_config(
    page_title="Fitness Video Analyzer",
    page_icon="💪",
    layout="wide"
)

# Set up OpenRouter API
def setup_openrouter_api() -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    """Set up the OpenRouter API connection with proper error handling."""
    # Get API key from session state (populated from sidebar or environment)
    api_key = st.session_state.api_key
    
    if not api_key:
        st.error("OPENROUTER_API_KEY not found. Please set it in the sidebar.")
        return None, None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://fitness-analyzer.example.com",  # Required by OpenRouter
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
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

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
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    if calculate_size:
        size_kb = len(buffer) / 1024
        return base64.b64encode(buffer).decode('utf-8'), size_kb
    
    return base64.b64encode(buffer).decode('utf-8'), None

# Analyze frames with OpenRouter
def analyze_with_openrouter_individual_frames(
    base_url: str, 
    headers: Dict[str, str], 
    video_file,
    frames_per_second: int = 4, 
    max_frames: int = 30, 
    model_name: str = "google/gemini-2.5-pro-exp-03-25"
) -> Tuple[str, List]:
    """Analyze video frames using OpenRouter API."""
    start_time = time.time()
    
    # Create a temporary file from the uploaded file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        tmpfile.write(video_file.getvalue())  # Use getvalue() instead of read()
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
            st.error(f"Error loading video: {str(e)}. Please check the file format.")
            return f"Error loading video: {str(e)}. Please check the file format.", []
        
        frames = extract_frames(tmpfile_path, fps=frames_per_second, max_frames=max_frames)
        
        if not frames:
            os.unlink(tmpfile_path)
            st.error("Error: No frames could be extracted from the video.")
            return "Error: No frames could be extracted from the video.", []
        
        # Display frame info
        st.info(f"Extracted {len(frames)} frames covering {len(frames)/frames_per_second:.2f} seconds of video")
        
        # Create a visual preview of frames (limit to 5 for display)
        if frames:
            preview_frames = frames[:5]
            cols = st.columns(len(preview_frames))
            for i, ((timestamp, frame), col) in enumerate(zip(preview_frames, cols)):
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
            "Be specific and detailed in your analysis. Use timestamps to reference specific moments in the exercise. "
            "Use biomechanical terms when appropriate, but also provide practical advice that a fitness enthusiast would understand."
        )
        
        OutputFormat = (
            "Respond with a detailed analysis in this format:\n"
            "- **Exercise identified**: [Name and detailed description of the exercise]\n"
            "- **Total repetition count**: [Number with timestamps for each complete rep]\n"
            "- **Tempo assessment**: [Detailed analysis of movement speed and rhythm]\n"
            "- **Form evaluation**: [Comprehensive assessment of posture, alignment, and technique]\n"
            "- **Recommendations**: [Specific, actionable advice for improvement]\n"
            "- **Reasoning**: [Your analysis process and observations]"
        )
        
        Content = [
            {"role": "system", "content": Role + "\n" + TaskPrompt + "\n" + OutputFormat}
        ]
        
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
        with st.spinner(f"Analyzing with {model_name}..."):
            try:
                # Debug info
                st.write(f"Making API request to {base_url}/chat/completions with model {model_name}")
                
                # Make the API request
                response = requests.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": model_name,
                        "messages": Content,
                        "temperature": 0.2,  # Slightly increased for more detailed output
                        "top_p": 0.1,  # Slightly increased
                        "max_tokens": 2000  # Increased for more detailed output
                    },
                    timeout=180  # Increased timeout
                )
                
                # Check for HTTP errors
                if response.status_code != 200:
                    st.error(f"API Error (Status {response.status_code}): {response.text}")
                    return f"API Error (Status {response.status_code}): {response.text}", Content
                
                # Parse the response
                try:
                    response_json = response.json()
                    st.write("API response received successfully")
                    
                    result = response_json['choices'][0]['message']['content']
                    processing_time = time.time() - start_time
                    st.success(f"Analysis completed in {processing_time:.2f} seconds")
                    
                    return result, Content
                except KeyError as e:
                    st.error(f"Unexpected API response format: {str(e)}")
                    st.write("Response content:", response.text)
                    return f"Unexpected API response format: {str(e)}", Content
                
            except requests.exceptions.Timeout:
                st.error("Request timed out. The video might be too complex or the service is busy.")
                return "Error: Request timed out. The video might be too complex or the service is busy.", Content
            except Exception as e:
                st.error(f"Error analyzing with OpenRouter: {str(e)}")
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
            value=st.session_state.api_key,
            help="Get your API key from https://openrouter.ai",
            key="api_key_input"
        )
        
        if api_key:
            # Update session state with API key
            st.session_state.api_key = api_key
            st.success("API key set!")
        
        st.markdown("---")
        st.markdown("Created by Your Name. v1.0.1")

# Streamlit app
def main():
    display_sidebar()
    
    st.title("💪 Fitness Video Analyzer")
    st.write("Upload a fitness video to analyze exercises, count reps, and evaluate form using AI.")
    
    # Define model options
    model_options = {
        "Free Models": [
            "google/gemini-2.5-pro-exp-03-25:free",
            "google/gemini-2.0-flash-thinking-exp:free",
            "meta-llama/llama-3.2-11b-vision-instruct:free",
            "qwen/qwen2.5-vl-32b-instruct:free"
        ],
        "Premium Models": [
            "anthropic/claude-3.7-sonnet:thinking",
            "anthropic/claude-3.7-sonnet",
            "openai/chatgpt-4o-latest",
            "microsoft/phi-4-multimodal-instruct"
        ]
    }
    
    # Create tabs for file upload and settings
    tab1, tab2 = st.tabs(["Upload Video", "Settings"])
    
    with tab1:
        # File uploader with better guidance
        uploaded_file = st.file_uploader(
            "Upload a fitness video",
            type=["mp4", "mov", "avi", "mkv"],
            help="For best results, use a clear video showing the full body movement"
        )
        
        # Display placeholder if no file uploaded
        if not uploaded_file:
            st.image("https://via.placeholder.com/640x360.png?text=Upload+a+fitness+video", use_column_width=True)
    
    with tab2:
        # Simplified settings
        col1, col2 = st.columns(2)
        
        with col1:
            # Model selection
            model_category = st.radio("Model Type", options=["Free Models", "Premium Models"])
            model_name = st.selectbox("Select Model", options=model_options[model_category])
            
            frames_per_second = st.slider(
                "Frames per second", 
                min_value=1, 
                max_value=10, 
                value=4
            )
        
        with col2:
            max_frames = st.slider(
                "Maximum frames to analyze", 
                min_value=5, 
                max_value=60, 
                value=20
            )
            
            min_dimension = st.slider(
                "Frame dimension (px)",
                min_value=200,
                max_value=800,
                value=400
            )
    
    # Analysis button
    if st.button("🔍 Analyze Video", type="primary", use_container_width=True, disabled=not uploaded_file):
        if not st.session_state.api_key:
            st.error("⚠️ Please enter your OpenRouter API key in the sidebar.")
        elif uploaded_file:
            # Check if file is actually a video
            try:
                # Reset the file position to the beginning
                uploaded_file.seek(0)
                
                # Create a results container
                results_container = st.container()
                
                with results_container:
                    st.markdown("## Analysis Results")
                    
                    # Check API key and setup
                    base_url, headers = setup_openrouter_api()
                    
                    if base_url and headers:
                        # Run analysis
                        response_text, content = analyze_with_openrouter_individual_frames(
                            base_url, headers, uploaded_file,
                            frames_per_second=frames_per_second,
                            max_frames=max_frames,
                            model_name=model_name
                        )
                        
                        if response_text.startswith("Error") or response_text.startswith("API Error"):
                            st.error(response_text)
                        else:
                            # Display results
                            st.markdown(response_text)
                            
                            # Add export options
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    "📄 Download Analysis",
                                    response_text,
                                    file_name="fitness_analysis.txt",
                                    mime="text/plain"
                                )
                            with col2:
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
                    else:
                        st.error("Cannot proceed without a valid API key.")
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                st.write("Please ensure you've uploaded a valid video file.")
        else:
            st.warning("Please upload a video file first.")

if __name__ == "__main__":
    main()