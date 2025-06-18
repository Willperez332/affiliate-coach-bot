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
    print("Generating strategic coaching report with GPT-4o...")
    
    top_winners, vanity_ref, dud_ref = find_best_references(style, GOLD_WINNERS, PUBLIC_WINNERS, VANITY_LOSERS, DUD_LOSERS)

    if not top_winners: 
        return "Error: Winners Library is empty. Use /learn to teach me first.", ""

    winners_text = json.dumps(top_winners, indent=2)
    vanity_text = json.dumps(vanity_ref, indent=2) if vanity_ref else "N/A"
    dud_text = json.dumps(dud_ref, indent=2) if dud_ref else "N/A"
    deconstruction_text = json.dumps(deconstruction, indent=2)

    system_prompt_main = """
You are CoachAI, an elite TikTok Shop performance coach. Your prime directive is to IMPROVE the user's existing video, not suggest a new one. All feedback must be actionable and grounded in the data provided. NEVER mention race, gender, or sensitive cultural topics unless the user's own video transcript contains them. Focus only on patterns from the WINNING_EXAMPLES.
"""

    user_prompt_main = f"""
**DATA LIBRARY:**
- WINNING EXAMPLES (A collection of top performers): {winners_text}
- VANITY LOSER (High Views, Low Sales): {vanity_text}
- DUD LOSER (Low Views, Low Sales): {dud_text}
- CREATOR'S VIDEO (Deconstruction): {deconstruction_text}
- INTENDED STYLE: {style}

**YOUR TASK (Follow this structure exactly):**
---
### **🧠 Quick Video Review**
*(Based on the library, what is the single biggest opportunity for improvement in the creator's video to drive more sales? Be direct and concise.)*

---
### **💡 Creative Brainstorm**
*(Based on the patterns in the WINNING EXAMPLES, provide the following):*
*   **Improved Hook:** Rewrite the creator's hook to be more powerful, using the 'spoken dialogue' style from the winning examples.
*   **Script Insight:** Pinpoint one specific sentence or moment in the middle of the creator's script that could be improved for better pacing or clarity. Provide a specific "before/after" suggestion.
*   **Style Tip:** Provide one general tip for the '{style}' video genre that is demonstrated in the winning examples and would directly help this video.
"""
    
    # THIS IS THE UPDATED SYSTEM PROMPT FOR THE EXPANDED VIEW
    system_prompt_expand = """
You are a direct, no-fluff video editor. Your only job is to provide AIDA-F feedback. For every weakness (🚩), you MUST provide a concrete, actionable fix (🛠️). This fix must be based on a STRATEGY or PATTERN you observe in the WINNING EXAMPLES. **IMPORTANT: Describe the strategy, do NOT mention the winning videos by name or specific content.** For example, instead of saying 'like the mother-daughter video', say 'by using a surprising age comparison to create intrigue'.
"""

    user_prompt_expand = f"""
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
        main_resp = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt_main}, {"role": "user", "content": user_prompt_main}],
            temperature=0.7
        )
        main_feedback = main_resp.choices[0].message.content.strip()
        main_feedback += "\n\n---\n### **Final Thought**\n*Keep pushing. The only difference between a good video and a viral video is a few small tweaks.*"

        expand_resp = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt_expand}, {"role": "user", "content": user_prompt_expand}],
            temperature=0.6
        )
        expand_feedback = expand_resp.choices[0].message.content.strip()

        return main_feedback, expand_feedback

    except Exception as e:
        print(f"Error generating report with GPT-4o: {e}")
        return f"An error occurred while generating the coaching report: {e}", ""
        
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
    1.  Analyze the video and transcribe it into a structured dialogue list. Each item in the list should be an object with a "speaker" and "dialogue" key.
    2.  The "speaker" can only be one of two options: 'Creator' (when the person on camera is talking) or 'Clip' (when dialogue is from an inserted movie clip, news report, etc.).
    3.  For each 'Clip', add a "source" key identifying the media if possible (e.g., "The Simpsons," "Joe Rogan Experience").

    **OUTPUT ONLY A STRUCTURED JSON OBJECT** with a single key "structured_transcript" that contains the list of dialogue objects.

    **GOOD EXAMPLE OUTPUT:**
    ```json
    {{
      "structured_transcript": [
        {{
          "speaker": "Creator",
          "dialogue": "You're telling me a movie from 2022 predicted this?"
        }},
        {{
          "speaker": "Clip",
          "source": "White Noise (Movie)",
          "dialogue": "You know what I don't understand is why they never found the doctors."
        }},
        {{
          "speaker": "Creator",
          "dialogue": "My son's had clean checkups every year since the day he was born."
        }}
      ]
    }}
    ```
    """

    try:
        response = await model.generate_content_async([prompt, video_file])
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        deconstruction_data = json.loads(cleaned_response)
        
        # Create the full deconstruction object for the rest of the bot
        full_deconstruction = {
            "transcript": transcript, # Keep the flat transcript for backwards compatibility
            "structured_transcript": deconstruction_data.get("structured_transcript", [])
        }
        print("Structured deconstruction successful.")
        return full_deconstruction
    except Exception as e:
        print(f"Error during structured deconstruction: {e}")
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

# --- DISCORD UI (V3 - CONTEXT AWARE) ---
class CoachingActions(discord.ui.View):
    def __init__(self, full_report, deconstruction, style):
        super().__init__(timeout=None)
        self.full_report = full_report
        self.deconstruction = deconstruction # Pass the whole object
        self.style = style

    @discord.ui.button(label="Show Full AIDA-F Breakdown", style=discord.ButtonStyle.secondary, emoji="🔬")
    async def show_full_report(self, button: discord.ui.Button, interaction: discord.Interaction):
        await interaction.response.send_message("Here is the detailed AIDA-F breakdown:", ephemeral=True)
        for chunk in split_message(self.full_report):
            await interaction.followup.send(chunk, ephemeral=True)

    @discord.ui.button(label="Rewrite My Script", style=discord.ButtonStyle.primary, emoji="✍️")
    async def rewrite_script(self, button: discord.ui.Button, interaction: discord.Interaction):
        await interaction.response.send_message("Got it! I'm starting the script rewrite. This might take a moment...", ephemeral=True)
        # Pass the full deconstruction object to the rewrite task
        asyncio.create_task(run_rewrite_task(interaction, self.deconstruction, self.style))

# --- SCRIPT REWRITING BRAIN (V3 - DIRECTOR'S CUT) ---
async def run_rewrite_task(interaction, deconstruction, style):
    print(f"Starting script rewrite for {style} style...")
    winner_ref, _, _ = find_best_references(style, GOLD_WINNERS, PUBLIC_WINNERS, VANITY_LOSERS, DUD_LOSERS, num_winners=1)
    
    system_prompt = "You are a world-class script doctor for viral videos. Your task is to rewrite the provided transcript into a structured, easy-to-record format, improving its pacing, clarity, and conversion potential based on the framework of a proven winning video."
    
    # This is the new, more powerful prompt
    user_prompt = f"""
**PROVEN WINNER'S FRAMEWORK (for reference):**
{json.dumps(winner_ref[0] if winner_ref else 'N/A', indent=2)}

**USER'S CURRENT SCRIPT (Structured Dialogue):**
{json.dumps(deconstruction.get('structured_transcript'), indent=2)}

**YOUR TASK:**
Rewrite the user's script into a structured format with clear framework labels.
1.  Analyze the user's script and the winner's framework.
2.  Rewrite the user's script, applying the winning patterns.
3.  Structure your output using these labels: `Hook:`, `Build-up:`, `Problem:`, `Solution/Proof:`, `CTA:`.
4.  **Crucially:** If the original script contained dialogue from a 'Clip' speaker, you MUST explicitly note where those clips should be played in the rewritten script. Use the format `(Play Clip: [Source])`. For example: `(Play Clip: The Simpsons)`.
5.  You must ONLY rewrite the dialogue where the speaker is 'Creator'. Preserve the 'Clip' dialogue.
6.  Output ONLY the final, rewritten script with the framework labels, ready for the creator to record.

**Example for a Conspiracy Video:**
Hook:
Why does nobody realize this Disney classic warned us about our future?

Build-up:
Listen to this scene, it's mind-blowing.
(Play Clip: Wall-E)
Now, it gets even creepier...

CTA:
...so if you see the orange box, grab it before it's gone for good.
"""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        rewritten_script = response.choices[0].message.content

        # Send the rewritten script in chunks
        await interaction.followup.send(f"### ✍️ **Your Rewritten Script (Director's Cut)**", ephemeral=True)
        for chunk in split_message(rewritten_script):
            await interaction.followup.send(chunk, ephemeral=True)

    except Exception as e:
        print(f"Error during script rewrite: {e}")
        await interaction.followup.send(f"An error occurred during the script rewrite: {e}", ephemeral=True)

# --- Background Task Runners (V3 - CONTEXT AWARE) ---
async def run_coaching_task(interaction, video_url, style, views):
    status_message = await interaction.followup.send("`[■□□□]` 🧠 Kicking off analysis...", wait=True)

    try:
        await status_message.edit(content="`[■■□□]` 🎥 Downloading & Deconstructing Video...")
        deconstruction = await process_video(video_url)
        if deconstruction.get("error"):
            await status_message.edit(content=f"Analysis failed during video processing: {deconstruction['details']}")
            return

        await status_message.edit(content="`[■■■□]` ✍️ Generating Your Coaching Report with GPT-4o...")
        main_feedback, expand_feedback = await generate_coaching_report(deconstruction, style, views)

        await status_message.edit(content="`[■■■■]` ✅ Analysis Complete! Your report is being delivered below.")
        
        thread = await interaction.channel.create_thread(
            name=f"Coaching for {interaction.user.display_name}",
            type=discord.ChannelType.public_thread
        )

        await thread.send(f"🎥 **Video:** {video_url}")

        if main_feedback:
            # Pass the full deconstruction object to the View
            view = CoachingActions(full_report=expand_feedback, deconstruction=deconstruction, style=style)
            await thread.send(main_feedback, view=view)

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
