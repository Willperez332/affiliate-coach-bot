# brain.py
import os
import json
import ast
import google.generativeai as genai

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
