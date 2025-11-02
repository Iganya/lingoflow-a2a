# agents/translation_agent.py
from typing import List, Optional
from uuid import uuid4
from groq import Groq
import os
import json
from .a2a_models import (
    A2AMessage, TaskResult, TaskStatus, Artifact,
    MessagePart, MessageConfiguration
)

class LingoFlowAgent:
    def __init__(self):
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is required")
        self.client = Groq(api_key=groq_api_key)
        self.system_prompt = (""" You are **TransBot**, a fast, accurate multilingual translation agent.
            ### CORE TASK
            1. **Detect** the language of the input text automatically (never ask the user).
            2. **Translate** the text to the **target language** specified by the user.
            3. **Always respond with a JSON object** in this **exact format**:

            ```json
            {
            "source_lang": "<ISO-639-1 code>",
            "target_lang": "<ISO-639-1 code>",
            "translation": "<translated text>"
            }
            ###RULES
            - Use ISO-639-1 two-letter codes (e.g., en, es, fr, de, zh, ja, ar, hi, pt, ru, ko, etc.).
            - If the user writes a full language name (e.g., "Spanish"), map it to the correct code.
            - Preserve all formatting: line breaks, bullet points, code blocks, emojis, punctuation.
            - If source and target are the same → "translation" = original text.
            - If language detection fails → assume source is en.
            - Never add explanations, comments, or extra text outside the JSON.
            - Never escape or wrap the JSON — output raw valid JSON only.
                        
        """)

    async def process_messages(
        self,
        messages: List[A2AMessage],
        context_id: Optional[str] = None,
        task_id: Optional[str] = None,
        config: Optional[MessageConfiguration] = None
    ) -> TaskResult:
        context_id = context_id or str(uuid4())
        task_id = task_id or str(uuid4())

        # Extract text and target language
        user_msg = messages[-1]
        text_to_translate = ""
        target_lang = "English"  # default

        for part in user_msg.parts:
            if part.kind == "text":
                raw = part.text.strip()
                # Extract target language if specified
                if " to " in raw.lower():
                    parts = raw.lower().split(" to ", 1)
                    text_to_translate = parts[0].strip()
                    target_lang = parts[1].strip().capitalize()
                else:
                    text_to_translate = raw
                break

        if not text_to_translate:
            raise ValueError("No text to translate")

        # Call Groq
        completion = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Translate to **{target_lang}**:\n\n\"\"\"\n{text_to_translate}\n\"\"\""}
            ],
            temperature=0.2,
            max_tokens=2048
        )
        response =  completion.choices[0].message.content
        result =  json.loads(response)
    
        translation = result.get("translation", text_to_translate)

        # Build response message
        response_msg = A2AMessage(
            role="agent",
            parts=[MessagePart(kind="text", text=translation)],
            taskId=task_id
        )

        # Artifacts
        artifacts = [
            Artifact(
                name="translation",
                parts=[MessagePart(kind="text", text=translation)]
            )
        ]

        # History
        history = [A2AMessage(
            role="agent",
            parts=[MessagePart(kind="text", text=translation)],
            taskId=task_id
        )]

        # Always ask for next input unless empty
        state = "working"
        if not user_msg:
            state = "input-required"
        elif translation:
            state = "completed"
        else:
            state = "failed"


        return TaskResult(
            id=task_id,
            contextId=context_id,
            status=TaskStatus(state=state, message=response_msg),
            artifacts=artifacts,
            history=history
        )

