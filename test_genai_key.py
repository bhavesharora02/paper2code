# test_genai_key.py
import os, google.generativeai as genai, sys

key = os.getenv("GEMINI_API_KEY")
if not key:
    print("No key in env.")
    sys.exit(1)

genai.configure(api_key=key)
try:
    model = genai.GenerativeModel("gemini-1.5-pro")
    resp = model.generate_content("Say 'hello' in JSON: ```json {\"hello\": \"world\"}```")
    # Try to print a short snippet safely
    text = getattr(resp, "text", None)
    if not text:
        try:
            text = resp.candidates[0].content.parts[0].text
        except Exception:
            text = str(resp)
    print("SUCCESS. Response preview:", (text or "")[:200])
except Exception as e:
    print("API test failed:", repr(e))
