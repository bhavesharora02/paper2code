# core/llm/gemini_client.py
"""
Backward-compatible Gemini client wrapper.

Provides:
 - class GeminiClient with:
    - generate(prompt, stream=False)  -> full text or generator (not needed by extractor)
    - json_call(prompt, max_retries=3) -> returns parsed JSON dict (this is used by extractor)
Uses google.generativeai.GenerativeModel under the hood.
"""

import os
import time
import json
import random
from typing import Any, Dict, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ServiceUnavailable, InternalServerError, TooManyRequests

load_dotenv()   # reads .env into os.environ

# Configure once (reads GEMINI_API_KEY from env)
_api_key = os.environ.get("GEMINI_API_KEY")
if not _api_key:
    # Don't raise here — raising later when calling will be clearer to user.
    pass
else:
    genai.configure(api_key=_api_key)


class GeminiClient:
    def __init__(self, model_name: str = "gemini-1.5-pro", temperature: float = 0.2):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment. Set it before calling Gemini.")
        self._model = genai.GenerativeModel(model_name)
        self._generation_config = {"temperature": temperature, "candidate_count": 1}

    # --- Compatibility method extractor expects ---
    def json_call(self, prompt: str, max_retries: int = 3, backoff_base: float = 0.8) -> Dict[str, Any]:
        """
        Call Gemini with prompt and expect a JSON response (fenced or plain).
        Retries on parse/API failures up to max_retries with exponential backoff.
        Returns parsed Python dict.
        """
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                text = self._call_once(prompt)
                json_text = self._extract_json_from_text(text)
                parsed = json.loads(json_text)
                return parsed
            except (json.JSONDecodeError, ValueError) as je:
                last_exc = je
                # JSON parse failed — try again with slight prompt nudging on last attempt
                if attempt < max_retries:
                    wait = backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    time.sleep(wait)
                    continue
                else:
                    raise RuntimeError(f"Failed to parse JSON from Gemini response after {max_retries} attempts. Last error: {je}\nRaw text:\n{text}") from je
            except (ServiceUnavailable, InternalServerError, TooManyRequests) as api_err:
                last_exc = api_err
                if attempt < max_retries:
                    wait = backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Gemini API error after {attempt} attempts: {api_err}") from api_err
            except Exception as e:
                last_exc = e
                if attempt < max_retries:
                    wait = backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Unhandled error calling Gemini: {e}") from e

        # if loop exits unexpectedly
        raise RuntimeError(f"Failed to get JSON from Gemini. Last error: {last_exc}")

    # --- lower-level call ---
    def _call_once(self, prompt: str) -> str:
        """
        Single call to the Gemini model. Returns full response text (string).
        """
        # Use generate_content which returns an object with .text or .candidates depending on client version
        resp = self._model.generate_content(prompt, generation_config=self._generation_config)
        # try a few common access patterns
        text = None
        if hasattr(resp, "text") and resp.text:
            text = resp.text
        else:
            try:
                # Some clients: resp.candidates[0].content.parts[0].text
                text = resp.candidates[0].content.parts[0].text
            except Exception:
                try:
                    # Another possible structure
                    text = resp.candidates[0].content[0].text
                except Exception:
                    text = str(resp)
        if text is None:
            raise RuntimeError(f"No text returned by Gemini. Full response object: {resp}")
        return str(text)

    # --- helpers ---
    @staticmethod
    def _extract_json_from_text(text: str) -> str:
        """
        Extract JSON from text. Supports fenced ```json ... ``` blocks, or raw {...} substrings.
        Returns the JSON substring (string) to be passed to json.loads.
        Raises ValueError if no plausible JSON is found.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Empty or non-string response from model")

        stripped = text.strip()

        # If fenced block exists, prefer it
        if stripped.startswith("```"):
            # find the first triple-fence content
            # e.g. ```json\n{...}\n```  OR ```\n{...}\n```
            parts = stripped.split("```")
            # parts[0] is empty if starts with ```
            for part in parts[1:]:
                part = part.strip()
                if not part:
                    continue
                # strip leading 'json' if present
                if part.lower().startswith("json"):
                    candidate = part[4:].strip()
                else:
                    candidate = part
                # try to find {...} inside candidate
                first = candidate.find("{")
                last = candidate.rfind("}")
                if first != -1 and last != -1 and last > first:
                    return candidate[first : last + 1].strip()
                # else try entire candidate
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    continue

        # otherwise try to find first {...} in the text
        first = stripped.find("{")
        last = stripped.rfind("}")
        if first != -1 and last != -1 and last > first:
            return stripped[first : last + 1].strip()

        # give up
        raise ValueError("No JSON object detected in model response")

    # Optional convenience: provide old-style name for backward-compatibility if someone imports GeminiJSON
    # So both `from core.llm.gemini_client import GeminiClient` and `... import GeminiJSON` will work
    GeminiJSON = None  # placeholder; set below


# Backwards compatibility alias
GeminiClient.GeminiJSON = GeminiClient
# also inject a module-level name GeminiJSON to satisfy imports
GeminiJSON = GeminiClient
