# bot.py - THE FINAL SENTIENT COACH (V29)
import discord
import os
import json
import uuid
from dotenv import load_dotenv
import time
import asyncio
import ast
from datetime import datetime, timezone
DATA_DIR = "/data"

# --- NEW IMPORTS ---
import openai
from pydub import AudioSegment

# --- Google Gen AI SDK & Cloud Tools ---
import google.generativeai as genai
import yt_dlp

# --- LOAD SECRETS & FINAL CONFIG ---
load_dotenv()

# --- NEW: INITIALIZE OPENAI CLIENT ---
client = openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') 

# --- INITIALIZE THE CORRECT SDK ---
genai.configure(api_key=GOOGLE_API_KEY)

# --- V29 UPGRADE: THE FOUR QUADRANT LIBRARY ---
# Replace your existing function with this one
def load_intelligence_library():
    gold_winners, public_winners, vanity_losers, dud_losers = [], [], [], []
    print("Loading Performance Quadrant Library...")
    
    # --- DEBUG LINE ---
    print(f"DEBUG: Attempting to read from directory: '{DATA_DIR}'")

    if not os.path.exists(DATA_DIR):
        print(f"Warning: Data directory '{DATA_DIR}' not found. Creating it.")
        os.makedirs(DATA_DIR)
        
    for filename in os.listdir(DATA_DIR):
        full_path = os.path.join(DATA_DIR, filename)
        if filename.startswith("library_gold_winner_"): gold_winners.append(json.load(open(full_path, 'r')))
        elif filename.startswith("library_public_winner_"): public_winners.append(json.load(open(full_path, 'r')))
        elif filename.startswith("library_vanity_loser_"): vanity_losers.append(json.load(open(full_path, 'r')))
        elif filename.startswith("library_dud_loser_"): dud_losers.append(json.load(open(full_path, 'r')))
    print(f"Library loaded with {len(gold_winners)} Gold, {len(public_winners)} Public, {len(vanity_losers)} Vanity, and {len(dud_losers)} Dud Losers.")
    return gold_winners, public_winners, vanity_losers, dud_losers
    
# --- INTELLIGENT MATCHING (V2) ---
def find_best_references(style, gold, public, vanity, duds, num_winners=3):
    """
    Finds a collection of the best winning examples and the most relevant losers.
    """
    # Combine all winners and filter by the requested style
    all_winners = gold + public
    style_winners = [v for v in all_winners if v.get("style", "").lower() == style.lower()]
    
    # Sort the winners by performance (views) and get the top N
    style_winners.sort(key=lambda v: v.get("views", 0), reverse=True)
    top_winners = style_winners[:num_winners]

    # If no style-specific winners, use the absolute best overall winners as a fallback
    if not top_winners:
        all_winners.sort(key=lambda v: v.get("views", 0), reverse=True)
        top_winners = all_winners[:num_winners]

    # Find the best loser examples for contrast
    best_vanity = next((v for v in vanity if v.get("style", "").lower() == style.lower()), vanity[0] if vanity else None)
    best_dud = next((v for v in duds if v.get("style", "").lower() == style.lower()), duds[0] if duds else None)
    
    return top_winners, best_vanity, best_dud
    
# --- BOT'S BRAIN (DECONSTRUCTION, STRATEGY, AND ANALYSIS) ---

async def deconstruct_and_summarize(video_file, performance_data):
    """
    Performs a deep, dual-brain analysis of the entire video and the specific hook.
    """
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
       - **hook_format:** Identify the format. Is it a 'visual_hook' (something shown), 'text_hook' (an overlay), 'auditory_hook' (a sound), or 'spoken_hook'?
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

# THIS IS THE CORRECT AND COMPLETE FUNCTION. REPLACE YOUR OLD ONE WITH THIS.
async def run_learning_task(interaction, video_url, style, views, sales_gmv, is_own_video):
    try:
        temp_filename = f"temp_video_{uuid.uuid4()}.mp4"
        uploaded_file = None
        try:
            print(f"Downloading video for learning: {video_url}")
            ydl_opts = {'outtmpl': temp_filename, 'format': 'mp4'}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([video_url])
            
            print(f"Uploading {temp_filename} for deep analysis...")
            uploaded_file = genai.upload_file(path=temp_filename)
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(10)
                uploaded_file = genai.get_file(name=uploaded_file.name)
            if uploaded_file.state.name != "ACTIVE":
                raise ValueError(f"File {uploaded_file.name} failed to process.")
            
            video_file = uploaded_file
            performance_data = {"views": views, "sales_gmv": sales_gmv}
            analysis_data = await deconstruct_and_summarize(video_file, performance_data)
        finally:
            if uploaded_file:
                genai.delete_file(name=uploaded_file.name)
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

        entry_data = {
            "name": f"New Entry ({style})",
            "style": style,
            "video_url": video_url,
            "views": views,
            "sales_gmv": sales_gmv,
            "analysis": analysis_data
        }
        
        file_prefix = "library_dud_loser_"
        if is_own_video:
            gmv_per_1k_views = (sales_gmv / views) * 1000 if views > 0 else 0
            if gmv_per_1k_views > 20 or sales_gmv > 10000:
                file_prefix = "library_gold_winner_"
            elif views > 500000:
                file_prefix = "library_vanity_loser_"
        else: 
            if views > 500000:
                file_prefix = "library_public_winner_"
        
        # This joins the /data directory with the generated filename
        file_name = os.path.join(DATA_DIR, f"{file_prefix}{uuid.uuid4()}.json")
        
        # This is the debug line that was missing from your log
        print(f"DEBUG: Attempting to save file to path: '{file_name}'")

        with open(file_name, 'w') as f:
            json.dump(entry_data, f, indent=2)
            
        global GOLD_WINNERS, PUBLIC_WINNERS, VANITY_LOSERS, DUD_LOSERS
        GOLD_WINNERS, PUBLIC_WINNERS, VANITY_LOSERS, DUD_LOSERS = load_intelligence_library()
        
        await interaction.followup.send(
            f"Success! I've deeply analyzed and saved the new performance data. My intelligence has been upgraded.",
            ephemeral=True)
            
    except Exception as e:
        print(f"Error in learning background task: {e}")
        await interaction.followup.send(
            "A critical error occurred during the learning process. Please check the logs.",
            ephemeral=True)
        
# --- BOT'S BRAIN (COACHING V2) ---
async def generate_coaching_report(deconstruction, style, views):
    print("Generating AIDA-F coaching report with GPT-4o...")
    
    top_winners, vanity_ref, dud_ref = find_best_references(style, GOLD_WINNERS, PUBLIC_WINNERS, VANITY_LOSERS, DUD_LOSERS)

    if not top_winners: 
        return "Error: Winners Library is empty. Use /learn to teach me first."

    winners_text = json.dumps(top_winners, indent=2)
    deconstruction_text = json.dumps(deconstruction, indent=2)
    
    system_prompt = """
You are CoachAI, an expert TikTok coach who gives clear, direct, and easy-to-understand advice. **Write like you're talking to a friend—use simple language that a high schooler could easily understand.** Avoid jargon and overly professional terms.

Your only job is to provide AIDA-F feedback. For every weakness (🚩), provide a concrete, actionable fix (🛠️) based on a STRATEGY or PATTERN from the winning examples. **Describe the strategy, do not cite the winning videos by name.** For example, instead of 'leverage social proof', say 'show, don't just tell, that people love this'.
"""

    user_prompt = f"""
**DATA LIBRARY:**
- WINNING EXAMPLES: {winners_text}
- CREATOR'S VIDEO: {deconstruction_text}

**YOUR TASK:**
Write ONLY the AIDA-F (Rapid Fire Feedback) section for this creator:
- Attention (Hook)
- Interest (Story/Problem)
- Desire (Solution/Proof)
- Action (CTA)
- Frame (Vibe)
For each section, provide two bullets: one "✅ Good" and one "🚩 Bad". For every "🚩 Bad", you must add a "🛠️ Fix:" with a specific, actionable suggestion based on a pattern from the WINNING EXAMPLES.
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.6
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error generating AIDA-F report with GPT-4o: {e}")
        return f"An error occurred while generating the AIDA-F report: {e}"
        
async def deconstruct_video(video_file, transcript):
    """
    Performs a structured deconstruction of the video, separating the Creator's
    dialogue from inserted clips.
    """
    print("Performing structured deconstruction with Gemini...")
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    prompt = f"""
    You are a master video analyst AI. Your task is to deconstruct the provided video into a structured format.

    **YOUR TASK:**
    1.  Analyze the video and transcribe it into a structured dialogue list. Each item in the list should be an object with "speaker", "dialogue", and "on_screen_text" keys.
    2.  The "speaker" can only be one of two options: 'Creator' or 'Clip'.
    3.  For each 'Clip', add a "source" key. **If the source is unknown, the value MUST be the string "Unknown Clip". Do not guess.**
    4.  At each timestamp, note any significant "on_screen_text" that appears. If there is no text, the value should be `null`.

    **OUTPUT ONLY A STRUCTURED JSON OBJECT** with a single key "structured_transcript" that contains the list of dialogue objects.
    """

    try:
        response = await model.generate_content_async([prompt, video_file])
        
        # --- THIS IS THE ROBUST PARSING LOGIC ---
        # 1. Clean the outer markdown fences
        cleaned_response = response.text.strip().replace("```json", "").replace("```python", "").replace("```", "")
        # 2. Safely replace JSON 'null' with Python 'None'
        cleaned_response = cleaned_response.replace('null', 'None')
        
        # 3. Use the flexible ast.literal_eval parser
        deconstruction_data = ast.literal_eval(cleaned_response)
        
        full_deconstruction = {
            "transcript": transcript,
            "structured_transcript": deconstruction_data.get("structured_transcript", [])
        }
        print("Structured deconstruction successful.")
        return full_deconstruction
    except Exception as e:
        print(f"Error during structured deconstruction: {e}")
        # Add the raw response to the error message for easier debugging
        print(f"--- PARSING FAILED --- Raw AI Response:\n{response.text}")
        return {"error": "Failed to deconstruct video.", "details": str(e)}
        
async def process_video(video_url):
    """
    Main pipeline for downloading, transcribing with Whisper, 
    and deconstructing visuals with Gemini.
    """
    temp_filename = f"temp_video_{uuid.uuid4()}.mp4"
    temp_audio_filename = f"temp_audio_{uuid.uuid4()}.mp3"
    uploaded_file = None
    transcript = ""

    try:
        # --- Download Video ---
        print(f"Downloading video from URL: {video_url}")
        ydl_opts = {'outtmpl': temp_filename, 'format': 'mp4'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([video_url])

        # --- 1. Extract Audio & Transcribe with Whisper ---
        print("Extracting audio for Whisper...")
        audio = AudioSegment.from_file(temp_filename, format="mp4")
        audio.export(temp_audio_filename, format="mp3")
        
        print("Transcribing with Whisper...")
        with open(temp_audio_filename, "rb") as audio_file:
            transcription_response = await client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
            transcript = transcription_response.text
        print("Whisper transcription successful.")

# --- 2. Upload Video for Visual Analysis with Gemini ---
        print(f"Uploading {temp_filename} for visual analysis...")

        # We'll retry up to 3 times in case of flakiness
        import random
        max_retries = 3
        for attempt in range(max_retries):
            try:
                uploaded_file = genai.upload_file(path=temp_filename)
                if not hasattr(uploaded_file, "name"):
                    raise ValueError("Upload failed — no file name returned from genai.upload_file().")
                break  # success
            except Exception as e:
                print(f"Attempt {attempt+1} to upload failed: {e}")
                if attempt < max_retries - 1:
                    sleep_time = 2 + random.random() * 3
                    print(f"Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                else:
                    raise RuntimeError(f"Failed to upload file to Gemini after {max_retries} attempts: {e}")

        # Poll until processing is done, also with error handling
        try:
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(10)
                uploaded_file = genai.get_file(name=uploaded_file.name)
        except Exception as e:
            raise RuntimeError(f"Failed to poll uploaded file status: {e}")

        if uploaded_file.state.name != "ACTIVE":
            raise ValueError(f"File {uploaded_file.name} failed to process. State: {uploaded_file.state.name}")

        # --- 3. Deconstruct Visuals with Gemini ---
        video_file = uploaded_file
        # Pass the Whisper transcript INTO the deconstruction function
        deconstruction = await deconstruct_video(video_file, transcript)
        return deconstruction

    finally:
        # --- Cleanup ---
        if uploaded_file:
            print(f"Deleting uploaded file: {uploaded_file.name}")
            genai.delete_file(name=uploaded_file.name)
        if os.path.exists(temp_filename):
            print(f"Deleting local temp video: {temp_filename}")
            os.remove(temp_filename)
        if os.path.exists(temp_audio_filename):
            print(f"Deleting local temp audio: {temp_audio_filename}")
            os.remove(temp_audio_filename)

# --- DISCORD UI (V4 - FINAL) ---
class CoachingActions(discord.ui.View):
    def __init__(self, deconstruction, style):
        super().__init__(timeout=None)
        self.deconstruction = deconstruction
        self.style = style

    @discord.ui.button(label="Generate Text Hooks", style=discord.ButtonStyle.success, emoji="✍️")
    async def suggest_text_hooks(self, button: discord.ui.Button, interaction: discord.Interaction):
        await interaction.response.send_message("Analyzing winning patterns to generate text hooks...", ephemeral=True)
        asyncio.create_task(run_text_hook_generation_task(interaction, self.deconstruction, self.style))

    @discord.ui.button(label="Rewrite as Pro Script", style=discord.ButtonStyle.primary, emoji="🎬")
    async def rewrite_script(self, button: discord.ui.Button, interaction: discord.Interaction):
        await interaction.response.send_message("Got it! I'm starting the production script rewrite. This can take a moment...", ephemeral=True)
        asyncio.create_task(run_rewrite_task(interaction, self.deconstruction, self.style))

# --- TEXT HOOK GENERATION BRAIN ---
async def run_text_hook_generation_task(interaction, deconstruction, style):
    print(f"Starting text hook generation for {style} style...")
    top_winners, _, _ = find_best_references(style, GOLD_WINNERS, PUBLIC_WINNERS, VANITY_LOSERS, DUD_LOSERS, num_winners=5)
    
    text_hook_examples = []
    if top_winners:
        for winner in top_winners:
            hook_analysis = winner.get("analysis", {}).get("hook_brain_analysis", {})
            if hook_analysis.get("hook_format") == "text_hook":
                text_hook_examples.append(hook_analysis.get("hook_text"))
            scenes = winner.get("analysis", {}).get("full_video_deconstruction", {}).get("scene_identification", [])
            for scene in scenes[:2]:
                if scene.get("on_screen_text"):
                    text_hook_examples.append(scene.get("on_screen_text"))
    
    user_hook_text = deconstruction.get("transcript", "").split('.')[0]

    system_prompt = "You are a viral TikTok marketing expert. Your job is to write short, punchy, on-screen text hooks that grab attention immediately. **Write like a creator, not a marketer. The hooks must be simple, direct, and easy to read in under 2 seconds.**"
    
    user_prompt = f"""
**WINNING TEXT HOOK EXAMPLES (for pattern inspiration):**
{json.dumps(text_hook_examples, indent=2)}

**USER'S SPOKEN HOOK (for context):**
"{user_hook_text}"

**YOUR TASK:**
Generate 3 unique, on-screen text hook ideas for the user's video.
- They must be short (ideally under 10 words).
- They must create curiosity or state a bold claim.
- They must be formatted as a simple, numbered list.
- Do NOT add any extra commentary. Just the hooks.
"""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.9
        )
        text_hooks = response.choices[0].message.content
        await interaction.followup.send(f"### ✍️ **3 Text Hooks to Test**\n\n{text_hooks}", ephemeral=True)

    except Exception as e:
        print(f"Error during text hook generation: {e}")
        await interaction.followup.send(f"An error occurred during text hook generation: {e}", ephemeral=True)

# --- SCRIPT REWRITING BRAIN (V6.1 - DEEP SEARCH FIX) ---
def find_scene_list(data):
    """
    Recursively searches a dictionary to find the first list, which is assumed
    to be the list of scenes.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key, value in data.items():
            found = find_scene_list(value)
            if found is not None:
                return found
    return None

async def run_rewrite_task(interaction, deconstruction, style):
    print(f"Starting Clean Script rewrite for {style} style...")
    winner_ref, _, _ = find_best_references(style, GOLD_WINNERS, PUBLIC_WINNERS, VANITY_LOSERS, DUD_LOSERS, num_winners=1)
    
    system_prompt = "You are an expert TikTok scriptwriter. Your task is to rewrite a video script to be more powerful and engaging, presenting it in a clean, scene-by-scene format. The output should be simple, clear, and easy for a creator to read and record."
    
    user_prompt = f"""
**PROVEN WINNER'S FRAMEWORK (for inspiration):**
{json.dumps(winner_ref[0] if winner_ref else 'N/A', indent=2)}

**USER'S CURRENT SCRIPT (Structured Dialogue):**
{json.dumps(deconstruction.get('structured_transcript'), indent=2)}

**YOUR TASK:**
Rewrite the user's script and structure it as a JSON list of scene objects.
1.  Each object in the list represents one scene and MUST have two keys: "scene_name" and "script_text".
2.  The "scene_name" should be a short label for the scene (e.g., "0-4s (Hook)").
3.  The "script_text" should contain the rewritten dialogue.
4.  **Crucially:** If a clip should be played, the "script_text" should simply be `(Play Clip: [Source])`.
5.  If on-screen text is essential, note it in parentheses, like `(On-screen Text: They LIED about this)`.
6.  Keep visual descriptions minimal. Focus on what needs to be said.

**OUTPUT ONLY A VALID JSON OBJECT containing the list of scenes.**

**PERFECT EXAMPLE OUTPUT:**
{{
  "script": [
    {{
      "scene_name": "0-4s (Hook)",
      "script_text": "I was sick for two years and doctors couldn't help. Then a stranger told me this..."
    }}
  ]
}}
"""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        raw_content = response.choices[0].message.content
        cleaned_content = raw_content.strip().replace("```json", "").replace("```", "")
        structured_script_data = ast.literal_eval(cleaned_content)
        
        # --- THIS IS THE NEW, ROBUST FINDER ---
        scenes = find_scene_list(structured_script_data)
        
        if not scenes:
            print(f"--- DEEP SEARCH FAILED --- Raw AI Response:\n{raw_content}")
            raise ValueError("Could not find the list of scenes in the AI's JSON response.")

        await interaction.followup.send(f"### 🎬 **Your New Script**", ephemeral=True)

        for scene in scenes:
            scene_name = scene.get("scene_name", "Unnamed Scene")
            script_text = scene.get("script_text", "N/A")

            embed = discord.Embed(
                description=script_text,
                color=discord.Color.from_rgb(49, 107, 242)
            )
            embed.set_author(name=f"🎬 {scene_name}")

            await interaction.followup.send(embed=embed, ephemeral=True)

    except Exception as e:
        print(f"Error during script rewrite: {e}")
        await interaction.followup.send(f"An error occurred during the script rewrite: {e}", ephemeral=True)
        
# --- TEXT HOOK GENERATION BRAIN ---
async def run_text_hook_generation_task(interaction, deconstruction, style):
    print(f"Starting text hook generation for {style} style...")
    top_winners, _, _ = find_best_references(style, GOLD_WINNERS, PUBLIC_WINNERS, VANITY_LOSERS, DUD_LOSERS, num_winners=5)
    
    # Extract text hooks from the winning examples
    text_hook_examples = []
    if top_winners:
        for winner in top_winners:
            hook_analysis = winner.get("analysis", {}).get("hook_brain_analysis", {})
            if hook_analysis.get("hook_format") == "text_hook":
                text_hook_examples.append(hook_analysis.get("hook_text"))
            # Also check the scene identification for on-screen text in the first few scenes
            scenes = winner.get("analysis", {}).get("full_video_deconstruction", {}).get("scene_identification", [])
            for scene in scenes[:2]: # Check first 2 scenes
                if scene.get("on_screen_text"):
                    text_hook_examples.append(scene.get("on_screen_text"))
    
    # Get the user's spoken hook for context
    user_hook_text = deconstruction.get("transcript", "").split('.')[0]

    system_prompt = "You are a viral TikTok marketing expert specializing in short, punchy, on-screen text hooks that stop the scroll. You think like a top-tier creator."
    
    user_prompt = f"""
**WINNING TEXT HOOK EXAMPLES (for pattern inspiration):**
{json.dumps(text_hook_examples, indent=2)}

**USER'S SPOKEN HOOK (for context):**
"{user_hook_text}"

**YOUR TASK:**
Generate 3 unique, on-screen text hook ideas for the user's video.
- They must be short (ideally under 10 words).
- They must create curiosity or state a bold claim.
- They must be formatted as a simple, numbered list.
- Do NOT add any extra commentary. Just the hooks.

**Bad Example:** "You should try making a hook that is more controversial."
**Good Example:**
1. They LIED about this one thing.
2. This is the #1 reason you're always tired.
3. My doctor was shocked when I showed him this.
"""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.9 # Higher temperature for more creativity
        )
        text_hooks = response.choices[0].message.content

        await interaction.followup.send(f"### ✍️ **3 Text Hooks to Test**\n\n{text_hooks}", ephemeral=True)

    except Exception as e:
        print(f"Error during text hook generation: {e}")
        await interaction.followup.send(f"An error occurred during text hook generation: {e}", ephemeral=True)


# --- Background Task Runners (V4 - FINAL) ---
async def run_coaching_task(interaction, video_url, style, views):
    status_message = await interaction.followup.send("`[■□□□]` 🧠 Kicking off analysis...", wait=True)

    try:
        await status_message.edit(content="`[■■□□]` 🎥 Downloading & Deconstructing Video...")
        deconstruction = await process_video(video_url)
        if deconstruction.get("error"):
            await status_message.edit(content=f"Analysis failed during video processing: {deconstruction['details']}")
            return

        await status_message.edit(content="`[■■■□]` ✍️ Generating Your AIDA-F Report...")
        aida_f_feedback = await generate_coaching_report(deconstruction, style, views)

        await status_message.edit(content="`[■■■■]` ✅ Analysis Complete! Your report is being delivered below.")
        
        thread = await interaction.channel.create_thread(
            name=f"Coaching for {interaction.user.display_name}",
            type=discord.ChannelType.public_thread
        )

        await thread.send(f"🎥 **Video:** {video_url}")

        if aida_f_feedback:
            # Create the view with the new buttons
            view = CoachingActions(deconstruction=deconstruction, style=style)
            # Send the AIDA-F feedback first, then the interactive buttons
            await thread.send("### 🔬 **AIDA-F Breakdown**")
            for chunk in split_message(aida_f_feedback):
                await thread.send(chunk)
            
            await thread.send("### 👇 **What's Next?**\nChoose an option below to get creative suggestions.", view=view)

    except Exception as e:
        print(f"Error in coaching background task: {e}")
        await status_message.edit(content=f"A critical error occurred: {e}")
        
# Utility function to chunk long messages
def split_message(text, chunk_size=1900):
    chunks = []
    while text:
        if len(text) <= chunk_size:
            chunks.append(text)
            break
        split_at = text.rfind('\n', 0, chunk_size)
        if split_at == -1:
            split_at = chunk_size
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip('\n')
    return chunks

# --- DISCORD MODALS ---
class LearningForm(discord.ui.Modal):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_item(discord.ui.InputText(label="Is this your own video? (yes/no)", placeholder="yes"))
        self.add_item(discord.ui.InputText(label="TikTok Video URL", placeholder="https://www.tiktok.com/..."))
        self.add_item(discord.ui.InputText(label="Video Style", placeholder="e.g., Conspiracy, Personal Story..."))
        self.add_item(discord.ui.InputText(label="Views", placeholder="e.g., 500000"))
        self.add_item(discord.ui.InputText(label="Sales ($ GMV) - ('0' if not your video)", placeholder="e.g., 550.50", style=discord.InputTextStyle.short))
    async def callback(self, interaction: discord.Interaction):
        await interaction.response.send_message("Got it! Deconstructing and deeply analyzing to learn. This may take a few minutes...", ephemeral=True)
        is_own_video = self.children[0].value.lower() == 'yes'
        video_url = self.children[1].value
        style = self.children[2].value
        try:
            views = int(self.children[3].value)
            sales_gmv = float(self.children[4].value.replace('$', ''))
        except ValueError:
            await interaction.followup.send("Please ensure Views and Sales are numbers only.", ephemeral=True)
            return
        asyncio.create_task(run_learning_task(interaction, video_url, style, views, sales_gmv, is_own_video))

class CoachingForm(discord.ui.Modal):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_item(discord.ui.InputText(label="TikTok Video URL", placeholder="https://www.tiktok.com/..."))
        self.add_item(discord.ui.InputText(label="Intended Video Style", placeholder="e.g., Conspiracy, Personal Story..."))
        self.add_item(discord.ui.InputText(label="Current Views (0 if not posted)", placeholder="e.g., 2500"))

    async def callback(self, interaction: discord.Interaction):
        # 1. Acknowledge INSTANTLY.
        await interaction.response.send_message("Got it! Your analysis is starting now. This can take a minute...", ephemeral=True)
        
        # 2. Get all the data from the form.
        video_url = self.children[0].value
        style = self.children[1].value
        try:
            views = int(self.children[2].value)
        except ValueError:
            await interaction.followup.send("Please enter a number for views.", ephemeral=True)
            return
            
        # 3. Start the long-running task in the background.
        asyncio.create_task(run_coaching_task(interaction, video_url, style, views))

# --- BOT SETUP AND COMMANDS ---
bot = discord.Bot()

@bot.event
async def on_ready():
    print("--- RUNNING BOT ---")
    print(f"{bot.user} is ready and online!")
    # Load the library from the persistent volume on startup
    global GOLD_WINNERS, PUBLIC_WINNERS, VANITY_LOSERS, DUD_LOSERS
    GOLD_WINNERS, PUBLIC_WINNERS, VANITY_LOSERS, DUD_LOSERS = load_intelligence_library()

# --- PASTE THE MISSING CODE HERE ---
@bot.slash_command(name="learn", description="Teach the AI new performance data.")
async def learn(ctx: discord.ApplicationContext):
    try: await ctx.send_modal(LearningForm(title="Teach CoachAI"))
    except discord.errors.NotFound: await ctx.respond("Timing issue. Please try again.", ephemeral=True)
# ------------------------------------

@bot.slash_command(name="coachme", description="Get your TikTok video analyzed by the AI Coach.")
async def coachme(ctx: discord.ApplicationContext):
    try: await ctx.send_modal(CoachingForm(title="Final AI Coach Submission"))
    except discord.errors.NotFound: await ctx.respond("Timing issue. Please try again.", ephemeral=True)

bot.run(DISCORD_TOKEN)
