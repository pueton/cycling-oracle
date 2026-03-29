import os, base64, json
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

app = FastAPI()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

class OracleResponse(BaseModel):
    reading: str
    prescription: str
    coffeeVerdict: str
    weather: str
    routeType: str

@app.post("/api/oracle", response_model=OracleResponse)
async def consult_oracle(
    file: UploadFile = File(...),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    lang: Optional[str] = Form("en")
):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured on server.")

    # Fetch weather if coordinates provided
    weather_context = ""
    weather_display = ""
    if lat is not None and lon is not None:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                w = await client.get(
                    f"https://api.open-meteo.com/v1/forecast"
                    f"?latitude={lat}&longitude={lon}"
                    f"&current=temperature_2m,weathercode,windspeed_10m,precipitation"
                    f"&wind_speed_unit=kmh"
                )
                wd = w.json()
                temp  = round(wd["current"]["temperature_2m"])
                wind  = round(wd["current"]["windspeed_10m"])
                rain  = wd["current"]["precipitation"]
                code  = wd["current"]["weathercode"]
                CODES = {0:"clear sky",1:"mainly clear",2:"partly cloudy",3:"overcast",
                         45:"foggy",51:"light drizzle",61:"light rain",63:"moderate rain",
                         65:"heavy rain",80:"showers",95:"thunderstorm"}
                desc = CODES.get(code, "unknown conditions")
                weather_context = f"Current weather: {temp}°C, {desc}, wind {wind} km/h, precipitation {rain} mm."
                weather_display = f"{temp}°C · {desc} · {wind} km/h wind"
        except Exception:
            weather_context = ""
            weather_display = ""

    image_bytes = await file.read()
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = file.content_type or "image/jpeg"

    prompt = (
        'You are The Espresso Oracle — a cycling coach and coffee snob who reads milk foam patterns. '
        f'{weather_context} '
        f'Respond in {"German" if lang == "de" else "English"}. '
        'Analyse this coffee foam image. Be dramatic but SHORT and CLEAR — no excessive jargon. '
        'If weather data is available, weave it naturally into the reading and prescription. '
        'Respond ONLY as valid JSON without any Markdown or extra text:\n'
        '{"reading":"2 short sentences max. What you see in the foam and what it means for the rider. Reference weather if available.",'
        '"prescription":"One clear sentence: what kind of ride today based on foam AND weather? e.g. easy spin, hard intervals, rest day.",'
        '"coffeeVerdict":"One short sentence on the coffee quality.",'
        '"routeType":"One of: easy, hilly, flat, long, intervals — best matching the prescription."}'
    )

    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 600,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": b64_image}},
            {"type": "text", "text": prompt}
        ]}]
    }
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

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

    return OracleResponse(
        reading=parsed["reading"],
        prescription=parsed["prescription"],
        coffeeVerdict=parsed["coffeeVerdict"],
        weather=weather_display,
        routeType=parsed.get("routeType", "moderate")
    )

app.mount("/", StaticFiles(directory="static", html=True), name="static")
