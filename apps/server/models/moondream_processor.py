import requests
import json
import base64
import logging
import re

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are AlmostHuman AI, a sophisticated holographic office receptionist.
Context: You can see the user through a camera. 

Rules:
- Be brief (under 15 words).
- Stay in character as a digital hologram.
- Use the 'Visual Context' to make the conversation feel real.
- If the visual context says 'no image', just greet them normally.
"""


class MoondreamProcessor:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.vision_model = "moondream"
        self.brain_model = "llama3.2"
        self.last_image_base64 = None

    async def set_image(self, image_data):
        self.last_image_base64 = base64.b64encode(image_data).decode("utf-8")
        return True

    def _get_visual_description(self):
        """Step 1: Ask Moondream what it sees"""
        if not self.last_image_base64:
            return "No image available."

        try:
            res = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.vision_model,
                    "prompt": "Describe the person in this image briefly.",
                    "images": [self.last_image_base64],
                    "stream": False,
                },
            )
            return res.json().get("response", "A person.")
        except:
            return "A visitor."

    def _smart_brain_stream(self, user_text, visual_context):
        """Step 2: Give that description to Llama 3.2 for a smart answer"""
        full_prompt = f"Visual Context: {visual_context}\nUser says: {user_text}"

        payload = {
            "model": self.brain_model,
            "system": SYSTEM_PROMPT,
            "prompt": full_prompt,
            "stream": True,
        }

        try:
            response = requests.post(self.ollama_url, json=payload, stream=True)
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    yield chunk.get("response", "")
                    if chunk.get("done"):
                        break
        except Exception as e:
            yield "System error."

    async def process_text_with_image(self, text, initial_chunks=3):
        # 1. Get the 'Eyes' to describe the scene
        visual_context = self._get_visual_description()
        logger.info(f"Hologram Eyes see: {visual_context}")

        # 2. Get the 'Brain' to generate the response
        streamer = self._smart_brain_stream(text, visual_context)

        initial_text = ""
        sentence_end_pattern = re.compile(r"[.!?]")

        # Buffer the first few words so the voice starts fast
        for chunk in streamer:
            initial_text += chunk
            if sentence_end_pattern.search(chunk) or len(initial_text) > 30:
                break

        def combined_generator():
            yield ""
            for chunk in streamer:
                yield chunk

        return combined_generator(), initial_text, True

    def update_history_with_complete_response(
        self, user_text, initial_response, remaining_text=None
    ):
        pass  # Simplified for performance
