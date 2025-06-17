# migrate_library.py
import os
import json
import asyncio
from brain import deconstruct_and_summarize # Import from the new brain file
from bot import DATA_DIR, genai, yt_dlp # Import only what's left

async def migrate_file(filename):
    """
    Takes one old V1 library file, re-analyzes its video,
    and saves the new V2 data structure to the persistent volume.
    """
    print(f"--- Processing V1 file: {filename} ---")
    try:
        # Load the old V1 data
        with open(filename, 'r') as f:
            v1_data = json.load(f)
        
        video_url = v1_data.get("video_url")
        if not video_url:
            print(f"SKIPPING: No video_url found in {filename}")
            return False

        # --- Re-run the full V2 analysis ---
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
            
            # Note: We don't need performance_data for this re-analysis
            video_file = uploaded_file
            analysis_data = await deconstruct_and_summarize(video_file, {})

        finally:
            if uploaded_file:
                genai.delete_file(name=uploaded_file.name)
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        
        if not analysis_data:
            raise ValueError("Analysis returned no data.")

        # --- Create the new V2 data structure ---
        v2_data = {
            "name": v1_data.get("name"),
            "style": v1_data.get("style"),
            "video_url": video_url,
            "views": v1_data.get("views"),
            "sales_gmv": v1_data.get("sales_gmv"),
            "analysis": analysis_data
        }

        # --- Save the new V2 file to the /data Volume ---
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
    
    # Find all old library files in the current directory (from GitHub)
    v1_files = [f for f in os.listdir('.') if f.startswith('library_') and f.endswith('.json')]
    
    if not v1_files:
        print("No V1 library files found to migrate.")
        return

    print(f"Found {len(v1_files)} V1 files to migrate.")
    
    success_count = 0
    failure_count = 0

    for filename in v1_files:
        if await migrate_file(filename):
            success_count += 1
        else:
            failure_count += 1
        await asyncio.sleep(5) # Add a small delay to avoid rate limiting

    print("\n===== MIGRATION COMPLETE =====")
    print(f"Successfully migrated: {success_count}")
    print(f"Failed to migrate:    {failure_count}")

if __name__ == "__main__":
    asyncio.run(main())
