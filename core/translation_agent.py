# agents/translation_agent.py
from typing import List, Optional
from uuid import uuid4
from groq import Groq
import os
import json
import re
from .response import _invalid_body_response
from .a2a_models import (
    A2AMessage, TaskResult, TaskStatus, Artifact,
    MessagePart, MessageConfiguration, ArtifactMessagePart
)

class LingoFlowAgent:
    def __init__(self):
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is required")
        
        self.client = Groq(api_key=groq_api_key)

        self.system_prompt = (""" You are **TransBot**, a fast, accurate multilingual translation agent.
                ### CORE TASK
                1. **Extract** the text to translate and the target language from any user input.
                2. **Detect** the source language of the extracted text automatically.
                3. **Translate** the text to the **target language**.
                4. **Always respond with a JSON object** in this **exact format**:

                ```json
                {
                "text_to_translate": "<original text>",
                "source_lang": "<ISO-639-1 code>",
                "target_lang": "<ISO-639-1 code>",
                "translation": "<translated text>"
                }
                EXTRACTION RULES
                Detect target language from phrases like: to <lang>, in <lang>, translate to <lang>, how do you say in <lang>, 
                              what is in <lang>, in <lang> how do one say, what do you think is <lang>, what is <lang>, etc.
                Use the last language indicator as the target.
                Extract only the text before that indicator as text_to_translate.
                User can call you Lingflow, translator etc, that should not be included in text to translate
                Preserve all formatting: punctuation, line breaks, emojis, quotes, code blocks, HTML.
                If no target language is found → default to "English".

                TRANSLATION RULES
                Use ISO-639-1 two-letter codes (e.g., en, es, fr, de, zh, ja, ar, hi, pt, ru, ko).
                Map full names: "Spanish" → es, "French" → fr, "Japanese" → ja, "Portuguese" → pt, etc.
                If source and target are the same → "translation" = original text.
                If source detection fails → assume en.
                Never add explanations, comments, or extra text outside the JSON.
                Output raw valid JSON only — no markdown, no code blocks, no wrapping.                 
        """)

        self.help_text = ("""Help Guide: Get Perfect Translations Every Time
            1. Put the target language at the end
            I love coding to Spanish → Works perfectly
            to Spanish I love coding → May fail
            2. Use to or in to indicate the target
            Hello world in French
            Say hello to Japanese
            3. Keep it simple — one translation per message
            Avoid: Life is beautiful to French and to Spanish
            4. No target language? It defaults to English
            Hola → Becomes: Hello
            5. Avoid double quotes in inline text or nested commands
            6. Still not working? Try these formats:
            What is 'I miss you' in Korean?""")
                

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
        raw_text = next(
            (part.data[-1]["text"] 
            for part in user_msg.parts
            if part.kind == "data"),
            None
        )
        if raw_text is None:
            raw_text = next(
                (part.text
                for part in user_msg.parts
                if part.kind == "text"),
                "Invalid Text"
            )

        # Return help guide for help input
        if raw_text.strip().lower() == 'help':
            return _invalid_body_response(request_id="", error=str(self.help_text))
        
        # Call Groq
        completion = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""Translate the following text: {raw_text}"""}
            ],
            temperature=0.2,
            max_tokens=2048
        )
        response =  completion.choices[0].message.content
        try:
            result =  json.loads(response)
        except Exception as e:
            # Fallback text for
            result = {
            "text_to_translate": "Invalid text",
            "source_lang": "en",
            "target_lang": "erpcn",
            "translation": "Invalid text"
            }

        translation = result.get("translation", "Kindly re enter the text again")
        text_to_translate = result.get("text_to_translate", "Invalid text",)
        source_code = result.get("source_lang", "en")
        target_code = result.get("target_lang", "en")

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
                parts=[ArtifactMessagePart(kind="text", text=translation)]
            ),
            Artifact(
                name="metadata",
                parts=[ArtifactMessagePart(kind="data", data={
                    "text_to_translate": text_to_translate,
                    "source_lang": source_code,
                    "target_lang": target_code
                })]
            )
        ]

        # History
        history = messages + [response_msg]

        # Setting state value
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
    
  



