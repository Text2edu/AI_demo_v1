import streamlit as st
import boto3
import json
import random
import time
from dotenv import load_dotenv
import concurrent.futures
import math
import urllib.request
import os
import tempfile
from moviepy import VideoFileClip, concatenate_videoclips, AudioFileClip
import uuid
from elevenlabs.client import ElevenLabs
from elevenlabs import save
import io

load_dotenv()

# Configuration
AWS_REGION = "us-east-1"
MODEL_ID_NOVA = "amazon.nova-reel-v1:0"
MODEL_ID_MISTRAL = "mistral.mistral-large-2402-v1:0"
S3_OUTPUT_BUCKET = "bedrock-video-generation-us-east-1-73hol2"

# Initialize clients
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
eleven_labs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

def generate_prompts(topic):
    user_msg = f"""
    You are an expert video scriptwriter specializing in creating educational content. 
    I will provide you with an educational topic. 
    Your task is to generate 10 detailed scene prompts for a 6-second video sequence and 10 corresponding audio prompts.

    **Topic:** {topic}

    **Output Format:**

    **Scene Prompts:**
    Scene 1: [Detailed description of length 75 words of the first 6-second scene]
    Scene 2: [Detailed description of length 75 words of the second 6-second scene]
    ...
    Scene 10: [Detailed description of length 75 words of the tenth 6-second scene]

    **Audio Prompts:**
    Audio 1: [Corresponding audio narration for Scene 1]
    Audio 2: [Corresponding audio narration for Scene 2]
    ...
    Audio 10: [Corresponding audio narration for Scene 10]

    Please ensure the scene prompts are visually descriptive and the audio prompts are clear and concise. 
    Maintain a consistent educational tone throughout.
    """

    messages = [{"role": "user", "content": [{"text": user_msg}]}]
    temperature = 0.0
    max_tokens = 2048

    params = {"modelId": MODEL_ID_MISTRAL,
              "messages": messages,
              "inferenceConfig": {"temperature": temperature,
                                  "maxTokens": max_tokens}}

    resp = bedrock_runtime.converse(**params)
    return resp["output"]["message"]["content"][0]["text"]

def parse_prompts(response):
    scene_prompts = []
    audio_prompts = []
    lines = response.split('\n')
    scene_section = False
    audio_section = False

    for line in lines:
        line = line.strip()
        if line.lower().startswith("**scene prompts:**"):
            scene_section = True
            audio_section = False
            continue
        elif line.lower().startswith("**audio prompts:**"):
            audio_section = True
            scene_section = False
            continue

        if scene_section and line.startswith("Scene"):
            scene_prompts.append(line.split(": ", 1)[1].strip())
        elif audio_section and line.startswith("Audio"):
            audio_prompts.append(line.split(": ", 1)[1].strip())

    return scene_prompts, audio_prompts

def generate_audio_narrations(audio_prompts):
    """Generate all audio narrations at once"""
    st.write("Generating audio narrations...")
    
    combined_text = ""
    for i, prompt in enumerate(audio_prompts):
        combined_text += f"{prompt}\n\n"
    
    try:
        # Use a temp directory for audio files
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "narrations.mp3")
        
        # Generate the audio - handle the generator correctly
        audio_generator = eleven_labs.text_to_speech.convert(
            text=combined_text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",  # Using the provided voice ID
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        
        # Collect all chunks from the generator into a single byte array
        audio_bytes = b""
        for chunk in audio_generator:
            if chunk:
                audio_bytes += chunk
        
        # Save the audio to a file
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
            
        st.success("Audio narrations generated successfully!")
        return audio_path
    except Exception as e:
        st.error(f"Error generating audio narrations: {e}")
        return None

def generate_video(video_prompt, video_number):
    combined_prompt = f"{video_prompt}"

    model_input = {
        "taskType": "TEXT_VIDEO",
        "textToVideoParams": {
            "text": combined_prompt
        },
        "videoGenerationConfig": {
            "durationSeconds": 6,
            "fps": 24,
            "dimension": "1280x720",
            "seed": random.randint(0, 2147483648)
        }
    }

    output_config = {"s3OutputDataConfig": {"s3Uri": f"s3://{S3_OUTPUT_BUCKET}"}}

    invocation = bedrock_runtime.start_async_invoke(
        modelId=MODEL_ID_NOVA,
        modelInput=model_input,
        outputDataConfig=output_config
    )

    invocation_arn = invocation["invocationArn"]
    s3_prefix = invocation_arn.split('/')[-1]
    s3_key = f"{s3_prefix}/output.mp4"
    https_url = f"https://{S3_OUTPUT_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
    st.write(f"Initiated video {video_number} generation.")

    SLEEP_TIME = 30
    while True:
        response = bedrock_runtime.get_async_invoke(invocationArn=invocation_arn)
        status = response["status"]
        st.write(f"Video {video_number} Status: {status}")
        if status != "InProgress":
            break
        time.sleep(SLEEP_TIME)

    if status == "Completed":
        st.write(f"Video {video_number} is ready at {https_url}")
        return https_url
    else:
        st.write(f"Video {video_number} generation failed with status: {status}")
        if "failureMessage" in response:
            st.write(f"Error message: {response['failureMessage']}")
        return None

def process_batch(batch_prompts, batch_number, total_batches):
    st.write(f"Processing batch {batch_number} of {total_batches}...")
    
    video_urls = []
    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i, (scene_index, scene_prompt) in enumerate(batch_prompts):
            future = executor.submit(generate_video, scene_prompt, scene_index + 1)
            futures.append((scene_index, future))

        completed_count = 0
        for scene_index, future in sorted(futures, key=lambda x: x[0]):
            url = future.result()
            video_urls.append((scene_index, url))
            completed_count += 1
            st.write(f"Batch {batch_number}: videos completed: {completed_count}/{len(futures)}")
    
    return video_urls
def download_video(url, output_path):
    """Download video from URL to a local path"""
    try:
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        st.error(f"Error downloading video: {e}")
        return False

def stitch_videos(video_urls, audio_path=None):
    """Stitch multiple videos together and optionally add audio"""
    # Create a temp directory
    temp_dir = tempfile.mkdtemp()
    video_paths = []
    
    # Download all videos
    with st.spinner("Downloading videos..."):
        for i, url in enumerate(video_urls):
            if url:
                video_path = os.path.join(temp_dir, f"video_{i+1}.mp4")
                if download_video(url, video_path):
                    video_paths.append(video_path)
    
    if not video_paths:
        st.error("No videos were successfully downloaded.")
        return None
    
    # Stitch videos
    with st.spinner("Stitching videos together..."):
        try:
            clips = [VideoFileClip(path) for path in video_paths]
            combined_clip = concatenate_videoclips(clips)
            
            # Add audio if provided
            if audio_path and os.path.exists(audio_path):
                audio_clip = AudioFileClip(audio_path)
                # Make sure audio doesn't exceed video length
                if audio_clip.duration > combined_clip.duration:
                    audio_clip = audio_clip.subclipped(0, combined_clip.duration)
                combined_clip = combined_clip.set_audio(audio_clip) if hasattr(combined_clip, 'set_audio') \
    else combined_clip.with_audio(audio_clip)
            
            # Generate a unique filename for the output
            output_path = os.path.join(temp_dir, f"combined_{uuid.uuid4().hex}.mp4")
            combined_clip.write_videofile(output_path, codec="libx264")
            
            # Close all clips to free memory
            for clip in clips:
                clip.close()
            if audio_path and os.path.exists(audio_path):
                audio_clip.close()
            combined_clip.close()
            
            return output_path
        except Exception as e:
            st.error(f"Error stitching videos: {e}")
            return None

# Initialize session state variables
if 'scene_prompts' not in st.session_state:
    st.session_state.scene_prompts = []
if 'audio_prompts' not in st.session_state:
    st.session_state.audio_prompts = []
if 'narration_audio' not in st.session_state:
    st.session_state.narration_audio = None
if 'all_video_urls' not in st.session_state:
    st.session_state.all_video_urls = []
if 'combined_video_path' not in st.session_state:
    st.session_state.combined_video_path = None
if 'combined_video_generated' not in st.session_state:
    st.session_state.combined_video_generated = False

st.title("Educational Video Generator")

topic = st.text_input("Enter the educational topic:")

generate_btn = st.button("Generate Video Series")

if generate_btn and topic:
    # Reset previous state when generating a new series
    st.session_state.combined_video_generated = False
    st.session_state.combined_video_path = None
    
    with st.spinner("Generating prompts..."):
        response = generate_prompts(topic)
        st.session_state.scene_prompts, st.session_state.audio_prompts = parse_prompts(response)

    if st.session_state.scene_prompts and st.session_state.audio_prompts:
        st.subheader("Scene Prompts:")
        for i, prompt in enumerate(st.session_state.scene_prompts):
            st.write(f"**Scene {i+1}:** {prompt}")

        st.subheader("Audio Prompts:")
        for i, prompt in enumerate(st.session_state.audio_prompts):
            st.write(f"**Audio {i+1}:** {prompt}")

        # Generate audio narrations
        st.session_state.narration_audio = generate_audio_narrations(st.session_state.audio_prompts)
        if st.session_state.narration_audio:
            st.audio(st.session_state.narration_audio, format="audio/mp3")
        
        st.subheader("Generating Videos:")
        
        # Process in batches of 5
        BATCH_SIZE = 5
        st.session_state.all_video_urls = []
        
        # Prepare indexed prompts
        indexed_prompts = list(enumerate(st.session_state.scene_prompts))
        total_batches = math.ceil(len(indexed_prompts) / BATCH_SIZE)
        
        # Process in batches
        for batch_idx in range(total_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(indexed_prompts))
            batch = indexed_prompts[start_idx:end_idx]
            
            batch_urls = process_batch(batch, batch_idx + 1, total_batches)
            st.session_state.all_video_urls.extend(batch_urls)

# If there are videos generated, display them
if st.session_state.all_video_urls:
    # Sort by original index
    st.session_state.all_video_urls.sort(key=lambda x: x[0])
    final_urls = [url for _, url in st.session_state.all_video_urls if url]
    
    # Display individual videos
    st.subheader("Individual Videos:")
    for i, (_, url) in enumerate(st.session_state.all_video_urls):
        if url:
            st.write(f"Video {i + 1}:")
            st.video(url)
            st.markdown(f'<a href="{url}" target="_blank">Open Video {i + 1} in new tab</a>', unsafe_allow_html=True)
            st.write("---")
    
    # Option to stitch videos together
    if len(final_urls) > 1:
        st.subheader("Combined Video")
        
        # Option to include narration
        include_narration = st.checkbox("Include audio narration in combined video", value=True)
        
        if st.button("Generate Combined Video"):
            st.session_state.combined_video_generated = True
            st.session_state.combined_video_path = stitch_videos(
                final_urls, 
                audio_path=st.session_state.narration_audio if include_narration else None
            )
        
        # Display combined video if available
        if st.session_state.combined_video_generated:
            if st.session_state.combined_video_path and os.path.exists(st.session_state.combined_video_path):
                st.video(st.session_state.combined_video_path)
                
                # Provide download link for combined video
                with open(st.session_state.combined_video_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Combined Video",
                        data=file,
                        file_name=f"combined_{topic.replace(' ', '_')}.mp4",
                        mime="video/mp4"
                    )
            else:
                st.error("Failed to create combined video.")