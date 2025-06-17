# migrate_library.py (Standalone Version)
import os
import json
import asyncio
import google.generativeai as genai
import yt_dlp
from dotenv import load_dotenv

# --- Standalone Configuration ---
# Load secrets from the .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not found.")
genai.configure(api_key=GOOGLE_API_KEY)

# Define the data directory directly
DATA_DIR = "/data"

# --- The Brain Function (copied directly here) ---
async def deconstruct_and_summarize(video_file, performance_data):
    """
    Performs a deep, dual-brain analysis of the entire video and the specific hook.
    """
    import ast # Keep import local to function
    print("Performing deep analysis for library...")
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    prompt = f"""
    You are a master viral video analyst. Your task is to perform a deep deconstruction of the provided video based on its performance data. You must analyze the video as a whole AND perform a specialized, deep analysis of the video's hook (the first 5-10 seconds).

    **PERFORMANCE DATA:**
    {json.dumps(performance_data, indent=2)}

    **YOUR TASK (Output a single structured dictionary):**

    **1. Full Video Deconstruction:**
       - **full_transcript:** Provide a complete, accurate transcript of the video.
       - **scene_identification:** Log the visual scenes with timestamps. Identify if a scene is 'user_talking_head', 'screen_recording', 'movie_clip', 'image_insert', etc. For movie/TV clips, identify the source if possible (e.g., 'The Simpsons', 'Wall-E').
       - **funnel_analysis:** In 2-3 sentences, describe how the video attempts to take a viewer from the hook to the final call-to-action. Analyze its 'organic feel' and how the creator established authority.
       - **core_lesson:** Based on the performance data, what is the single most important strategic lesson this video teaches us?

    **2. Specialized Hook Analysis ("Hook Brain"):**
       - **hook_text:** Transcribe the first 5-10 seconds of spoken dialogue.
       - **hook_format:** Identify the format. Is it a 'visual_hook', 'text_hook', 'auditory_hook', or 'spoken_hook'?
       - **hook_type:** Is the hook a 'question', a 'bold_statement', a 'story_lead_in', or a 'controversial_claim'?
       - **tonality:** Describe the vocal tonality of the hook (e.g., "Urgent and conspiratorial," "Calm and authoritative," "Excited and personal").
       - **emotional_trigger:** What primary emotion is the hook designed to trigger? (e.g., 'Curiosity', 'Fear', 'Anger', 'Hope', 'FOMO').

    **OUTPUT ONLY A STRUCTURED PYTHON DICTIONARY LITERAL that contains all of these keys.**
    """
    response = await model.generate_content_async([prompt, video_file])
    cleaned_response = response.text.strip().replace("```json", "").replace("```python", "").replace("```", "")
    print("Deep analysis successful.")
    try:
        return ast.literal_eval(cleaned_response)
    except (ValueError, SyntaxError) as e:
        print(f"--- PARSING FAILED ---")
        print(f"Error: {e}")
        print(f"Raw AI Response:\n{cleaned_response}")
        raise ValueError("Failed to parse the AI's response into a dictionary.") from e

# --- The Migration Logic ---
async def migrate_file(filename, original_dir):
    print(f"--- Processing V1 file: {filename} ---")
    try:
        v1_filepath = os.path.join(original_dir, filename)
        with open(v1_filepath, 'r') as f:
            v1_data = json.load(f)
        
        video_url = v1_data.get("video_url")
        if not video_url:
            print(f"SKIPPING: No video_url found in {filename}")
            return False

        temp_filename = f"temp_migration_video_{os.path.basename(filename)}.mp4"
        uploaded_file = None
        analysis_data = None
        
        try:
            print(f"Downloading video for migration: {video_url}")
            ydl_opts = {'outtmpl': temp_filename, 'format': 'mp4'}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            print(f"Uploading {temp_filename} for deep analysis...")
            uploaded_file = genai.upload_file(path=temp_filename)
            while uploaded_file.state.name == "PROCESSING":
                await asyncio.sleep(10)
                uploaded_file = genai.get_file(name=uploaded_file.name)
            if uploaded_file.state.name != "ACTIVE":
                raise ValueError(f"File {uploaded_file.name} failed to process.")
            
            video_file = uploaded_file
            analysis_data = await deconstruct_and_summarize(video_file, {})

        finally:
            if uploaded_file:
                genai.delete_file(name=uploaded_file.name)
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        
        if not analysis_data:
            raise ValueError("Analysis returned no data.")

        v2_data = {
            "name": v1_data.get("name"),
            "style": v1_data.get("style"),
            "video_url": video_url,
            "views": v1_data.get("views"),
            "sales_gmv": v1_data.get("sales_gmv"),
            "analysis": analysis_data
        }

        new_filename = os.path.join(DATA_DIR, os.path.basename(filename))
        print(f"SUCCESS: Saving V2 data to {new_filename}")
        with open(new_filename, 'w') as f:
            json.dump(v2_data, f, indent=2)
        
        return True

    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR: Failed to migrate {filename}. Reason: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return False

async def main():
    print("===== STARTING LIBRARY MIGRATION =====")
    original_dir = "/app" # In Railway, the repo files are in /app
    
    v1_files = [f for f in os.listdir(original_dir) if f.startswith('library_') and f.endswith('.json')]
    
    if not v1_files:
        print("No V1 library files found to migrate in /app directory.")
        return

    print(f"Found {len(v1_files)} V1 files to migrate.")
    
    success_count = 0
    failure_count = 0

    for filename in v1_files:
        if await migrate_file(filename, original_dir):
            success_count += 1
        else:
            failure_count += 1
        await asyncio.sleep(5)

    print("\n===== MIGRATION COMPLETE =====")
    print(f"Successfully migrated: {success_count}")
    print(f"Failed to migrate:    {failure_count}")

# This makes the script runnable
if __name__ == "__main__":
    # Check if the /data directory exists, if not, create it.
    if not os.path.exists(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' not found. Creating it for migration.")
        os.makedirs(DATA_DIR)
    asyncio.run(main())
