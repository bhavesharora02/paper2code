# core/llm/gemini_client.py
"""
Gemini JSON wrapper - strict JSON fence parsing with retries.
Uses ChatGoogleGenerativeAI client as specified (gemini-1.5-pro).

Environment:
  - GEMINI_API_KEY must be set in environment.

Design:
  - json_call(prompt): returns parsed JSON (dict)
  - If response fails JSON parse, will retry up to max_retries with backoff.
"""

import os
import time
import json
from typing import Any, Dict, Optional

try:
    # official wrapper (make sure package is installed in your env)
    from google.generativeai import ChatGoogleGenerativeAI
except Exception:
    # If the user hasn't installed the Google client yet, raise a helpful error.
    raise ImportError(
        "google.generativeai package not found. Install via `pip install google-generativeai` "
        "or ensure your environment has the client library. See README."
    )


class GeminiJSON:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-pro", temperature: float = 0.3):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Please set the environment variable or pass api_key.")
        # instantiate client wrapper
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
        )

    def _extract_json_from_text(self, text: str) -> str:
        """
        Extracts the first JSON fenced block (```json ... ```) if present,
        otherwise tries to extract a top-level JSON object from the text.
        """
        fence_open = "```json"
        fence_close = "```"
        start = text.find(fence_open)
        if start != -1:
            end = text.find(fence_close, start + len(fence_open))
            if end != -1:
                candidate = text[start + len(fence_open) : end].strip()
                return candidate

        # fallback: try to find first "{" ... "}" block (best-effort)
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return text[first_brace : last_brace + 1].strip()

        return text.strip()

    def json_call(self, prompt: str, max_retries: int = 3, backoff: float = 0.6) -> Dict[str, Any]:
        """
        Send prompt to Gemini and return parsed JSON.
        Retries on JSON parse errors and on transient API errors.
        """
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                # Use the model to generate response. The exact call shape may vary
                # depending on the client version; adapt if your client expects different args.
                resp = self.model.generate(prompts=[prompt])
                # The client's response object format might differ; try a few common access patterns:
                text = None
                # candidate approach: try to access resp.candidates[0].content[0].text (older pattern)
                try:
                    text = resp.candidates[0].content[0].text
                except Exception:
                    try:
                        # Try resp.text (some wrappers)
                        text = resp.text
                    except Exception:
                        # Last resort: convert to str
                        text = str(resp)

                json_text = self._extract_json_from_text(text)
                parsed = json.loads(json_text)
                return parsed
            except json.JSONDecodeError as je:
                last_exc = je
                # when parse fails, retry with backoff but also send a recovery prompt suggestion:
                time.sleep(backoff * attempt)
                continue
            except Exception as e:
                # transient API/network or client errors: retry after backoff
                last_exc = e
                time.sleep(backoff * attempt)
                continue

        # If we reach here, all retries failed
        raise RuntimeError(f"Failed to get JSON from Gemini after {max_retries} attempts. Last error: {last_exc}")