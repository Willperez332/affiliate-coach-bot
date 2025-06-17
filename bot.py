# bot.py - THE FINAL SENTIENT COACH (V29)
import discord
import os
import json
import uuid
from dotenv import load_dotenv
import time
import asyncio
import ast
from datetime import datetime

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
def load_intelligence_library():
    gold_winners, public_winners, vanity_losers, dud_losers = [], [], [], []
    print("Loading Performance Quadrant Library...")
    for filename in os.listdir('.'):
        if filename.startswith("library_gold_winner_"): gold_winners.append(json.load(open(filename, 'r')))
        elif filename.startswith("library_public_winner_"): public_winners.append(json.load(open(filename, 'r')))
        elif filename.startswith("library_vanity_loser_"): vanity_losers.append(json.load(open(filename, 'r')))
        elif filename.startswith("library_dud_loser_"): dud_losers.append(json.load(open(filename, 'r')))
    print(f"Library loaded with {len(gold_winners)} Gold, {len(public_winners)} Public, {len(vanity_losers)} Vanity, and {len(dud_losers)} Dud Losers.")
    return gold_winners, public_winners, vanity_losers, dud_losers
GOLD_WINNERS, PUBLIC_WINNERS, VANITY_LOSERS, DUD_LOSERS = load_intelligence_library()

# --- INTELLIGENT MATCHING ---
def find_best_references(style, gold, public, vanity, duds):
    best_winner = next((v for v in gold if v.get("style", "").lower() == style.lower()), gold[0] if gold else None)
    if not best_winner: best_winner = next((v for v in public if v.get("style", "").lower() == style.lower()), public[0] if public else None)
    best_vanity = next((v for v in vanity if v.get("style", "").lower() == style.lower()), vanity[0] if vanity else None)
    best_dud = next((v for v in duds if v.get("style", "").lower() == style.lower()), duds[0] if duds else None)
    return best_winner, best_vanity, best_dud

# --- HOOK LEARNING SYSTEM ---
HOOKS_FILE = "hooks.json"

def load_hooks():
    if not os.path.exists(HOOKS_FILE):
        return []
    with open(HOOKS_FILE, 'r') as f:
        return json.load(f)

def find_similar_hooks(style, max_results=3):
    """
    Return up to `max_results` hooks matching the style, sorted by highest views.
    """
    hooks = load_hooks()
    # Filter by style (case-insensitive)
    filtered = [h for h in hooks if h.get("style", "").lower() == style.lower()]
    # Sort by views (desc)
    filtered.sort(key=lambda h: h.get("views", 0), reverse=True)
    return filtered[:max_results]

def save_hook(hook_data):
    hooks = load_hooks()
    hooks.append(hook_data)
    with open(HOOKS_FILE, 'w') as f:
        json.dump(hooks, f, indent=2)

# --- BOT'S BRAIN (DECONSTRUCTION, STRATEGY, AND ANALYSIS) ---
async def deconstruct_and_summarize(video_file, performance_data):
    """Performs a deep analysis and distills the core strategy."""
    print("Performing deep analysis for library...")
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    prompt = f"""
    You are a master viral video analyst. Your task is to deconstruct the provided video and then, based on the performance data, distill its core strategic lesson.

    **PERFORMANCE DATA:**
    {json.dumps(performance_data, indent=2)}

    **YOUR TASK:**
    1.  **Deconstruct the video:** Provide a full transcript, a visual log with timestamps, and a pacing/energy analysis.
    2.  **Distill the Core Lesson:** Based on the deconstruction and the performance data, answer this question in 1-2 powerful sentences: **"What is the single most important strategic lesson this video teaches us?"** (e.g., "This video teaches us that a highly controversial hook can generate massive views but fails to convert because it attracts the wrong audience," or "This video proves that a vulnerable, personal story hook builds enough trust to drive high sales even with moderate views.")

    **OUTPUT ONLY A STRUCTURED PYTHON DICTIONARY LITERAL** with the keys: "deconstruction" (containing the full analysis) and "core_lesson".
    """
    response = await model.generate_content_async([prompt, video_file])
    
    # --- NEW, MORE ROBUST PARSING METHOD ---
    # First, remove the markdown fences.
    cleaned_response = response.text.strip().replace("```json", "").replace("```python", "").replace("```", "")
    
    print("Deep analysis successful.")
    
    # Use ast.literal_eval, which is safer and handles quote variations.
    try:
        return ast.literal_eval(cleaned_response)
    except (ValueError, SyntaxError) as e:
        print(f"--- PARSING FAILED ---")
        print(f"Error: {e}")
        print(f"Raw AI Response:\n{cleaned_response}")
        raise ValueError("Failed to parse the AI's response into a dictionary.") from e

async def generate_coaching_report(deconstruction, style, views):
    print("Generating strategic coaching report with GPT-4o...")
    winner_ref, vanity_ref, dud_ref = find_best_references(style, GOLD_WINNERS, PUBLIC_WINNERS, VANITY_LOSERS, DUD_LOSERS)

    if not winner_ref: return "Error: Winners Library is empty. Use /learn to teach me first."
    
    winner_text = json.dumps(winner_ref, indent=2)
    vanity_text = json.dumps(vanity_ref, indent=2) if vanity_ref else "N/A"
    dud_text = json.dumps(dud_ref, indent=2) if dud_ref else "N/A"
    deconstruction_text = json.dumps(deconstruction, indent=2)

    # Note the change to OpenAI's required message format
    system_prompt = "You are CoachAI, an elite TikTok Shop performance marketing coach. Your analysis is sharp, direct, and backed by a library of real performance data. Your primary mandate is to help creators create videos that are **both viral AND profitable.**"
    
    user_prompt = f"""
**YOUR KNOWLEDGE BASE:**
- Proven Winner From Market Data: {winner_text}
- Vanity Performer: {vanity_text}
- Underperformer: {dud_text}
- Creator's Video Deconstruction: {deconstruction_text}
- Intended Video Style: {style}

**YOUR TASK:**
Reply in a simple, easy-to-skim format.
For each section, use MAX 2 sentences per bullet. Focus only on what matters most.
No fluff, no repetition, no jargon—just real, creator-friendly feedback.

---
### 🧠 **Quick Video Review**
- **First Impression:** (1 short sentence: What's the #1 thing holding this video back from going viral & converting?)

---
### **AIDA-F (Rapid Fire Feedback)**

**Attention (Hook):**
- ✅ Good: (Best thing about the hook)
- 🚩 Bad: (Main issue, keep it short)
- 🛠️ Fix: (One actionable, specific change. Reference winning video if relevant.)

**Interest (Story/Problem):**
- ✅ Good:
- 🚩 Bad:
- 🛠️ Fix:

**Desire (Solution/Proof):**
- ✅ Good:
- 🚩 Bad:
- 🛠️ Fix:

**Action (CTA):**
- ✅ Good:
- 🚩 Bad:
- 🛠️ Fix:

**Frame (Vibe):**
- ✅ Good:
- 🚩 Bad:
- 🛠️ Fix:

---
### **Creative Brainstorm**
- **Improved Hook:** (Rewrite the creator’s hook—scroll-stopping, fits their style)
- **Script Insight:** (Pinpoint *where* in the video to add a new line or fix something—be specific, e.g. “Right after the first testimonial, add…”)
- **Style Tip:** (Share one creative trick that top {style} videos use)

---
### **Final Thought**
(Finish with one unique, motivational quote for creators. Never repeat.)
"""


    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
            coaching_feedback = response.choices[0].message.content

    # --- Add Hook Brain Suggestions ---
    similar_hooks = find_similar_hooks(style)
    if similar_hooks:
        hook_examples = "\n".join([
            f"- \"{h['hook_text']}\" (Views: {h['views']})"
            for h in similar_hooks if h.get('hook_text')
        ])
        hook_tip = (
            f"\n\n### 🔥 Proven Hook Examples ({style.title()}):\n"
            f"{hook_examples}\n"
            "Steal inspiration from these: notice their tone, structure, and first few seconds."
        )
    else:
        hook_tip = ""

    return coaching_feedback + hook_tip

    except Exception as e:
        print(f"Error generating report with GPT-4o: {e}")
        return f"An error occurred while generating the coaching report with GPT-4o: {e}"

async def deconstruct_video(video_file, transcript):
    """
    Performs visual-only deconstruction, using a provided transcript and
    generates visual log with timestamps to support time-specific insights.
    If no meaningful improvements are found, it will skip unnecessary suggestions.
    """
    print("Performing visual-only deconstruction with Gemini...")
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    prompt = f"""
    You are a video analysis AI. A highly accurate transcript has been provided for context. 
    Your task is to ONLY analyze the visual elements of the video and include timestamps for key moments.

    **TRANSCRIPT:**
    ---
    {transcript}
    ---

    **YOUR TASK:**
    1. Create a visual log with accurate timestamps. Each entry should include timestamp, camera angle, on-screen text, and notable visual features.
    2. Identify and return the most impactful moment in the video based on pacing, expression, or visual delivery. Include its timestamp and a short reason why it stands out.
    3. ONLY provide a script improvement suggestion IF there is a clear opportunity to enhance clarity, tone, or persuasion in a specific moment. Otherwise, skip it.

    **OUTPUT FORMAT:**
    ```json
    {{
        "visual_log": [
            {{"timestamp": "00:05", "description": "Text overlay: \"Wall-E was right...\", medium close-up of creator with shocked expression."}},
            {{"timestamp": "00:12", "description": "Insert of Wall-E footage, grainy texture with yellow filter."}}
        ],
        "highlight_moment": {{
            "timestamp": "00:12",
            "reason": "This is where the creator shifts tone and inserts the viral Wall-E moment, anchoring attention."
        }},
        "script_suggestion": {{
            "timestamp": "00:13",
            "suggestion": "Right after this, add a line like: 'This is the part they don't want you to see.'"
        }}
    }}
    OR if no changes are needed:
    ```json
    {{
        "visual_log": [...],
        "highlight_moment": {{...}},
        "script_suggestion": null
    }}
    ```
    """

    try:
        response = await model.generate_content_async([prompt, video_file])
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        visual_log_data = json.loads(cleaned_response)
        full_deconstruction = {
            "transcript": transcript,
            "visual_log": visual_log_data.get("visual_log", []),
            "highlight_moment": visual_log_data.get("highlight_moment", {}),
            "script_suggestion": visual_log_data.get("script_suggestion", None)
        }
        print("Visual-only deconstruction successful.")
        return full_deconstruction
    except Exception as e:
        print(f"Error during visual deconstruction: {e}")
        return {"error": "Failed to deconstruct video visuals.", "details": str(e)}


async def process_video_for_report(video_url, style, views):
    """Main pipeline for the /coachme command."""
    # This is a simplified deconstruction for speed, since the full analysis is the goal.
    temp_filename = f"temp_video_{uuid.uuid4()}.mp4"
    uploaded_file = None
    try:
        print(f"Downloading video for coaching: {video_url}")
        ydl_opts = {'outtmpl': temp_filename, 'format': 'best[ext=mp4]/best'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([video_url])
        print(f"Uploading {temp_filename}...")
        uploaded_file = genai.upload_file(path=temp_filename)
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(10)
            uploaded_file = genai.get_file(name=uploaded_file.name)
        if uploaded_file.state.name != "ACTIVE": raise ValueError(f"File {uploaded_file.name} failed to process.")
        video_file = uploaded_file
        
        # We still run the simple deconstruction here for the coaching report
        deconstruction = await deconstruct_video(video_file)
        return await generate_coaching_report(deconstruction, style)
    finally:
        if uploaded_file: genai.delete_file(name=uploaded_file.name)
        if os.path.exists(temp_filename): os.remove(temp_filename)

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

# --- Background Task Runners ---

async def run_coaching_task(interaction, video_url, style, views):
    # --- Start by sending the first status message ---
    status_message = await interaction.followup.send("`[■□□□]` 🧠 Kicking off analysis...", wait=True)

    try:
        # --- Update status before the heavy lifting ---
        await status_message.edit(content="`[■■□□]` 🎥 Downloading & Deconstructing Video...")
        deconstruction = await process_video(video_url)

        # Check for errors from the processing step
        if deconstruction.get("error"):
            await status_message.edit(content=f"Analysis failed during video processing: {deconstruction['details']}")
            return

        # --- Update status again before the AI report generation ---
        await status_message.edit(content="`[■■■□]` ✍️ Generating Your Coaching Report with GPT-4o...")
        analysis_report = await generate_coaching_report(deconstruction, style, views)

        # --- Final success message and report delivery ---
        await status_message.edit(content="`[■■■■]` ✅ Analysis Complete! Your report is being delivered below.")
        
        thread = await interaction.channel.create_thread(
            name=f"Coaching for {interaction.user.display_name}",
            type=discord.ChannelType.public_thread
        )
        if analysis_report:
            video_header = f"🎥 **Video:** {video_url}\n\n"
            full_report = video_header + analysis_report

            # Split into <=2000 character chunks, *without* breaking words mid-line
            chunks = []
            while len(full_report) > 0:
                if len(full_report) <= 2000:
                    chunks.append(full_report)
                    break
                split_at = full_report.rfind('\n', 0, 2000)
                if split_at == -1:
                    split_at = 2000
                chunks.append(full_report[:split_at])
                full_report = full_report[split_at:].lstrip('\n')
            
            for chunk in chunks:
                await thread.send(chunk)
    except Exception as e:
        print(f"Error in coaching background task: {e}")
        # Update the status message with the error
        await status_message.edit(content=f"A critical error occurred: {e}")

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
            "deconstruction": analysis_data["deconstruction"],
            "core_lesson": analysis_data["core_lesson"]
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
        
        file_name = f"{file_prefix}{uuid.uuid4()}.json"
        print(f"--- SAVING NEW DATA ({file_prefix}) ---")
        with open(file_name, 'w') as f:
            json.dump(entry_data, f, indent=2)
            
        global GOLD_WINNERS, PUBLIC_WINNERS, VANITY_LOSERS, DUD_LOSERS
        GOLD_WINNERS, PUBLIC_WINNERS, VANITY_LOSERS, DUD_LOSERS = load_intelligence_library()
        
# ---- Hook Extraction and Save (HOOK BRAIN) ----
        hook_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "style": style,
            "video_url": video_url,
            "views": views,
            "sales_gmv": sales_gmv,
            "hook_text": None,
            "hook_analysis": None,
            "full_entry": copy.deepcopy(entry_data)
        }

        # Try to auto-extract hook info if available
        decon = analysis_data.get("deconstruction", {})
        transcript = decon.get("transcript", "")
        if transcript:
            # Extract the first 10 seconds or first 1-2 sentences (can adjust logic)
            lines = transcript.split('\n')
            hook_candidate = ""
            for line in lines:
                if len(hook_candidate) < 300:
                    hook_candidate += line.strip() + " "
                else:
                    break
            hook_data["hook_text"] = hook_candidate.strip()

        # Optionally save more hook analysis if your deconstruction provides it
        hook_data["hook_analysis"] = decon.get("visual_log", [])  # or other info

        # Save to hooks.json
        hooks = load_hooks()
        hooks.append(hook_data)
        with open(HOOKS_FILE, 'w') as f:
            json.dump(hooks, f, indent=2)
        
        await interaction.followup.send(
            f"Success! I've deeply analyzed and saved the new performance data. My intelligence has been upgraded.",
            ephemeral=True)
    except Exception as e:
        print(f"Error in learning background task: {e}")
        await interaction.followup.send(
            "A critical error occurred during the learning process. Please check the logs.",
            ephemeral=True)

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
async def on_ready(): print(f"{bot.user} is ready and online!")
@bot.slash_command(name="learn", description="Teach the AI new performance data.")
async def learn(ctx: discord.ApplicationContext):
    try: await ctx.send_modal(LearningForm(title="Teach CoachAI"))
    except discord.errors.NotFound: await ctx.respond("Timing issue. Please try again.", ephemeral=True)
@bot.slash_command(name="coachme", description="Get your TikTok video analyzed by the AI Coach.")
async def coachme(ctx: discord.ApplicationContext):
    try: await ctx.send_modal(CoachingForm(title="Final AI Coach Submission"))
    except discord.errors.NotFound: await ctx.respond("Timing issue. Please try again.", ephemeral=True)
bot.run(DISCORD_TOKEN)
