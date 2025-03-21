from fastapi import FastAPI, BackgroundTasks, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional
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
import uuid
import pusher
from moviepy import VideoFileClip, concatenate_videoclips, AudioFileClip
from elevenlabs.client import ElevenLabs
import io

app = FastAPI(title="Educational Video Generator API")

load_dotenv()

# Configuration
AWS_REGION = "us-east-1"
MODEL_ID_NOVA = "amazon.nova-reel-v1:0"
MODEL_ID_MISTRAL = "mistral.mistral-large-2402-v1:0"
S3_OUTPUT_BUCKET = "bedrock-video-generation-us-east-1-73hol2"

# Initialize clients
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
eleven_labs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Initialize Pusher
pusher_client = pusher.Pusher(
    app_id=os.getenv('PUSHER_APP_ID', ''),
    key=os.getenv('PUSHER_KEY', ''),
    secret=os.getenv('PUSHER_SECRET', ''),
    cluster=os.getenv('PUSHER_CLUSTER', 'ap2'),
    ssl=True
)

# Data Models
class TopicRequest(BaseModel):
    topic: str
    session_id: str

class VideoGenerationRequest(BaseModel):
    session_id: str
    include_narration: bool = True

class PromptResponse(BaseModel):
    session_id: str
    scene_prompts: List[str]
    audio_prompts: List[str]

class VideoResponse(BaseModel):
    session_id: str
    video_urls: List[str]
    status: str

# Session storage (in a production app, use Redis or a database)
sessions = {}

def send_update(session_id, event, data):
    """Send real-time update via Pusher"""
    try:
        pusher_client.trigger(f'session-{session_id}', event, data)
    except Exception as e:
        print(f"Pusher error: {e}")

def generate_prompts(topic, session_id):
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

    send_update(session_id, "status_update", {"message": "Generating prompts..."})
    
    params = {"modelId": MODEL_ID_MISTRAL,
              "messages": messages,
              "inferenceConfig": {"temperature": temperature,
                                  "maxTokens": max_tokens}}

    try:
        resp = bedrock_runtime.converse(**params)
        return resp["output"]["message"]["content"][0]["text"]
    except Exception as e:
        send_update(session_id, "error", {"message": f"Error generating prompts: {str(e)}"})
        raise e

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

def generate_audio_narrations(audio_prompts, session_id):
    """Generate all audio narrations at once"""
    send_update(session_id, "status_update", {"message": "Generating audio narrations..."})
    
    combined_text = ""
    for i, prompt in enumerate(audio_prompts):
        combined_text += f"{prompt}\n\n"
    
    try:
        # Use a temp directory for audio files
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, f"narrations_{session_id}.mp3")
        
        # Generate the audio
        audio_generator = eleven_labs.text_to_speech.convert(
            text=combined_text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
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
            
        send_update(session_id, "audio_complete", {"message": "Audio narrations completed!"})
        return audio_path
    except Exception as e:
        send_update(session_id, "error", {"message": f"Error generating audio narrations: {str(e)}"})
        raise e

def generate_video(video_prompt, video_number, session_id):
    combined_prompt = f"{video_prompt}"
    
    send_update(session_id, "status_update", {
        "message": f"Starting video {video_number} generation...",
        "video_number": video_number
    })

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

    try:
        invocation = bedrock_runtime.start_async_invoke(
            modelId=MODEL_ID_NOVA,
            modelInput=model_input,
            outputDataConfig=output_config
        )

        invocation_arn = invocation["invocationArn"]
        s3_prefix = invocation_arn.split('/')[-1]
        s3_key = f"{s3_prefix}/output.mp4"
        https_url = f"https://{S3_OUTPUT_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        
        send_update(session_id, "status_update", {
            "message": f"Initiated video {video_number} generation.",
            "video_number": video_number
        })

        SLEEP_TIME = 30
        while True:
            response = bedrock_runtime.get_async_invoke(invocationArn=invocation_arn)
            status = response["status"]
            send_update(session_id, "status_update", {
                "message": f"Video {video_number} Status: {status}",
                "video_number": video_number,
                "status": status
            })
            if status != "InProgress":
                break
            time.sleep(SLEEP_TIME)

        if status == "Completed":
            send_update(session_id, "video_complete", {
                "message": f"Video {video_number} is ready",
                "video_number": video_number,
                "url": https_url
            })
            return https_url
        else:
            failure_message = response.get('failureMessage', 'Unknown error')
            send_update(session_id, "error", {
                "message": f"Video {video_number} generation failed: {failure_message}",
                "video_number": video_number
            })
            return None
    except Exception as e:
        send_update(session_id, "error", {
            "message": f"Error generating video {video_number}: {str(e)}",
            "video_number": video_number
        })
        return None

def process_batch(batch_prompts, batch_number, total_batches, session_id):
    send_update(session_id, "status_update", {
        "message": f"Processing batch {batch_number} of {total_batches}...",
        "batch": batch_number,
        "total_batches": total_batches
    })
    
    video_urls = []
    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i, (scene_index, scene_prompt) in enumerate(batch_prompts):
            future = executor.submit(generate_video, scene_prompt, scene_index + 1, session_id)
            futures.append((scene_index, future))

        completed_count = 0
        for scene_index, future in sorted(futures, key=lambda x: x[0]):
            url = future.result()
            video_urls.append((scene_index, url))
            completed_count += 1
            send_update(session_id, "batch_progress", {
                "message": f"Batch {batch_number}: videos completed: {completed_count}/{len(futures)}",
                "batch": batch_number,
                "completed": completed_count,
                "total": len(futures)
            })
    
    return video_urls

def download_video(url, output_path, session_id):
    """Download video from URL to a local path"""
    try:
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        send_update(session_id, "error", {"message": f"Error downloading video: {str(e)}"})
        return False

def stitch_videos(video_urls, audio_path, session_id):
    """Stitch multiple videos together and optionally add audio"""
    # Create a temp directory
    temp_dir = tempfile.mkdtemp()
    video_paths = []
    
    # Download all videos
    send_update(session_id, "status_update", {"message": "Downloading videos..."})
    for i, url in enumerate(video_urls):
        if url:
            video_path = os.path.join(temp_dir, f"video_{i+1}.mp4")
            if download_video(url, video_path, session_id):
                video_paths.append(video_path)
    
    if not video_paths:
        send_update(session_id, "error", {"message": "No videos were successfully downloaded."})
        return None
    
    # Stitch videos
    send_update(session_id, "status_update", {"message": "Stitching videos together..."})
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
        combined_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
        # Close all clips to free memory
        for clip in clips:
            clip.close()
        if audio_path and os.path.exists(audio_path):
            audio_clip.close()
        combined_clip.close()
        
        # Save output path to S3 or other storage for access
        # For now, we'll use a local path but in production you'd upload to cloud storage
        public_url = f"/videos/{os.path.basename(output_path)}"
        send_update(session_id, "combined_complete", {
            "message": "Combined video is ready",
            "url": public_url
        })
        
        return output_path
    except Exception as e:
        send_update(session_id, "error", {"message": f"Error stitching videos: {str(e)}"})
        return None

async def process_topic_async(topic: str, session_id: str):
    try:
        # Generate prompts
        response = generate_prompts(topic, session_id)
        scene_prompts, audio_prompts = parse_prompts(response)
        
        # Store in session
        sessions[session_id] = {
            "topic": topic,
            "scene_prompts": scene_prompts,
            "audio_prompts": audio_prompts,
            "video_urls": [],
            "narration_audio": None,
            "combined_video_path": None
        }
        
        # Send prompts to client
        send_update(session_id, "prompts_ready", {
            "scene_prompts": scene_prompts,
            "audio_prompts": audio_prompts
        })
        
        # Generate audio narrations
        narration_audio = generate_audio_narrations(audio_prompts, session_id)
        sessions[session_id]["narration_audio"] = narration_audio
        
        # Prepare indexed prompts
        indexed_prompts = list(enumerate(scene_prompts))
        BATCH_SIZE = 5
        total_batches = math.ceil(len(indexed_prompts) / BATCH_SIZE)
        
        all_video_urls = []
        # Process in batches
        for batch_idx in range(total_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(indexed_prompts))
            batch = indexed_prompts[start_idx:end_idx]
            
            batch_urls = process_batch(batch, batch_idx + 1, total_batches, session_id)
            all_video_urls.extend(batch_urls)
            
        # Sort by original index
        all_video_urls.sort(key=lambda x: x[0])
        final_urls = [url for _, url in all_video_urls if url]
        sessions[session_id]["video_urls"] = final_urls
        
        send_update(session_id, "videos_ready", {
            "message": "All videos are ready",
            "urls": final_urls
        })
        
    except Exception as e:
        send_update(session_id, "error", {"message": f"Error processing topic: {str(e)}"})

async def stitch_videos_async(session_id: str, include_narration: bool):
    try:
        session = sessions.get(session_id)
        if not session:
            send_update(session_id, "error", {"message": "Session not found"})
            return
        
        video_urls = session.get("video_urls", [])
        if not video_urls:
            send_update(session_id, "error", {"message": "No videos available for stitching"})
            return
        
        audio_path = session.get("narration_audio") if include_narration else None
        
        combined_path = stitch_videos(video_urls, audio_path, session_id)
        if combined_path:
            sessions[session_id]["combined_video_path"] = combined_path
            # In a real app, you'd generate a secure URL to this file or upload to cloud storage
    except Exception as e:
        send_update(session_id, "error", {"message": f"Error stitching videos: {str(e)}"})

# API Endpoints
@app.post("/api/generate-prompts", response_model=PromptResponse)
async def create_prompts(request: TopicRequest, background_tasks: BackgroundTasks):
    session_id = request.session_id
    topic = request.topic
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Start processing in background
    background_tasks.add_task(process_topic_async, topic, session_id)
    
    return {"session_id": session_id, "scene_prompts": [], "audio_prompts": []}

@app.post("/api/combine-videos", response_model=VideoResponse)
async def combine_videos(request: VideoGenerationRequest, background_tasks: BackgroundTasks):
    session_id = request.session_id
    
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Start video stitching in background
    background_tasks.add_task(stitch_videos_async, session_id, request.include_narration)
    
    return {
        "session_id": session_id,
        "video_urls": sessions[session_id].get("video_urls", []),
        "status": "processing"
    }

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return {
        "session_id": session_id,
        "topic": session.get("topic", ""),
        "scene_prompts": session.get("scene_prompts", []),
        "audio_prompts": session.get("audio_prompts", []),
        "video_urls": session.get("video_urls", []),
        "has_narration": session.get("narration_audio") is not None,
        "has_combined_video": session.get("combined_video_path") is not None
    }

# Serve static files - in production use nginx/cloudfront for this
@app.get("/videos/{video_name}")
async def get_video(video_name: str):
    # This is just a placeholder - in production you'd use proper file serving
    # or return a signed URL to cloud storage
    from fastapi.responses import FileResponse
    
    for session_id, session in sessions.items():
        combined_path = session.get("combined_video_path")
        if combined_path and os.path.basename(combined_path) == video_name:
            return FileResponse(combined_path)
    
    raise HTTPException(status_code=404, detail="Video not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)