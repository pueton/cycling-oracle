import os, base64, json
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

class OracleResponse(BaseModel):
    reading: str
    prescription: str
    coffeeVerdict: str

@app.post("/api/oracle", response_model=OracleResponse)
async def consult_oracle(file: UploadFile = File(...)):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured on server.")
    image_bytes = await file.read()
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = file.content_type or "image/jpeg"
    prompt = (
        'You are The Espresso Oracle — a world-class road cycling coach and insufferable coffee snob who reads milk foam patterns. '
        'Analyse this coffee foam image. Respond ONLY as valid JSON without any Markdown or extra text:\n'
        '{"reading":"Mystical cycling foam reading using terms like watts, FTP, KOM, peloton. Dramatic and pretentious. 2-3 sentences.",'
        '"prescription":"One specific ride prescription: zone, terrain, duration.",'
        '"coffeeVerdict":"One snobbish sentence on the espresso quality based on the foam."}'
    )
    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 600,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": b64_image}},
            {"type": "text", "text": prompt}
        ]}]
    }
    headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    data = resp.json()
    raw = "".join(b.get("text", "") for b in data["content"]).replace("```json","").replace("```","").strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=500, detail="Could not parse model response: " + raw)
    return OracleResponse(reading=parsed["reading"], prescription=parsed["prescription"], coffeeVerdict=parsed["coffeeVerdict"])

app.mount("/", StaticFiles(directory="static", html=True), name="static")
