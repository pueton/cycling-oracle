import os, base64, json, httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
STRAVA_CLIENT_ID   = os.environ.get("STRAVA_CLIENT_ID", "")
STRAVA_CLIENT_SECRET = os.environ.get("STRAVA_CLIENT_SECRET", "")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:10000")

# ── Models ──
class OracleResponse(BaseModel):
    reading: str
    prescription: str
    coffeeVerdict: str
    weather: str
    routeType: str

class StravaAthlete(BaseModel):
    id: int
    firstname: str
    lastname: str
    profile: str
    city: Optional[str] = ""

class StravaSegment(BaseModel):
    id: int
    name: str
    distance: float
    avg_grade: float
    kom: str
    entry_count: int
    url: str

# ── Oracle ──
@app.post("/api/oracle", response_model=OracleResponse)
async def consult_oracle(
    file: UploadFile = File(...),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    lang: Optional[str] = Form("en")
):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured.")

    weather_context = ""
    weather_display = ""
    if lat is not None and lon is not None:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                w = await client.get(
                    f"https://api.open-meteo.com/v1/forecast"
                    f"?latitude={lat}&longitude={lon}"
                    f"&current=temperature_2m,weathercode,windspeed_10m,precipitation"
                )
                wd = w.json()
                temp  = round(wd["current"]["temperature_2m"])
                wind  = round(wd["current"]["windspeed_10m"])
                code  = wd["current"]["weathercode"]
                CODES = {0:"clear sky",1:"mainly clear",2:"partly cloudy",3:"overcast",
                         45:"foggy",51:"light drizzle",61:"light rain",63:"moderate rain",
                         65:"heavy rain",80:"showers",95:"thunderstorm"}
                desc = CODES.get(code, "unknown")
                weather_context = f"Current weather: {temp}°C, {desc}, wind {wind} km/h."
                weather_display = f"{temp}°C · {desc} · {wind} km/h"
        except Exception:
            pass

    image_bytes = await file.read()
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = file.content_type or "image/jpeg"

    prompt = (
        f'You are The Espresso Oracle — a cycling coach and coffee snob who reads milk foam. '
        f'{weather_context} '
        f'Respond in {"German" if lang == "de" else "English"}. '
        f'Analyse this coffee foam. Be dramatic but SHORT and CLEAR. '
        f'Respond ONLY as valid JSON without Markdown:\n'
        f'{{"reading":"2 short sentences. What you see and what it means. Reference weather if available.",'
        f'"prescription":"One clear sentence: what kind of ride today?",'
        f'"coffeeVerdict":"One short snobbish sentence on the coffee quality.",'
        f'"routeType":"One of: easy, hilly, flat, long, intervals"}}'
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
        raise HTTPException(status_code=500, detail="Could not parse response: " + raw)

    return OracleResponse(
        reading=parsed["reading"],
        prescription=parsed["prescription"],
        coffeeVerdict=parsed["coffeeVerdict"],
        weather=weather_display,
        routeType=parsed.get("routeType", "easy")
    )

# ── Strava OAuth ──
@app.get("/api/strava/auth")
async def strava_auth():
    from urllib.parse import quote
    redirect_uri = quote(f"{BASE_URL}/api/strava/callback", safe="")
    url = (
        f"https://www.strava.com/oauth/authorize"
        f"?client_id={STRAVA_CLIENT_ID}"
        f"&redirect_uri={redirect_uri}"
        f"&response_type=code"
        f"&approval_prompt=auto"
        f"&scope=read,activity:read"
    )
    return RedirectResponse(url)

@app.get("/api/strava/callback")
async def strava_callback(code: str, request: Request):
    async with httpx.AsyncClient() as client:
        resp = await client.post("https://www.strava.com/oauth/token", data={
            "client_id": STRAVA_CLIENT_ID,
            "client_secret": STRAVA_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code"
        })
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Strava auth failed")
    data = resp.json()
    token = data["access_token"]
    athlete = data["athlete"]
    # Redirect back to frontend with token + athlete info
    name = f"{athlete['firstname']} {athlete['lastname']}"
    profile = athlete.get("profile_medium", athlete.get("profile", ""))
    return RedirectResponse(
        f"/?strava_token={token}&strava_name={name}&strava_profile={profile}&strava_id={athlete['id']}"
    )

# ── Strava Segments near location ──
@app.get("/api/strava/segments")
async def get_segments(
    lat: float, lon: float,
    route_type: str = "easy",
    token: str = ""
):
    if not token:
        raise HTTPException(status_code=401, detail="No Strava token")

    # Bounding box ~10km around location
    delta = 0.09
    bounds = f"{lat-delta},{lon-delta},{lat+delta},{lon+delta}"

    # Map route type to min/max grade
    grade_map = {
        "flat":      (None, 2),
        "easy":      (None, 4),
        "hilly":     (3, None),
        "long":      (None, 5),
        "intervals": (4, None),
    }
    min_grade, max_grade = grade_map.get(route_type, (None, None))

    params = {"bounds": bounds, "activity_type": "riding"}
    if min_grade: params["min_grade"] = min_grade
    if max_grade: params["max_grade"] = max_grade

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            "https://www.strava.com/api/v3/segments/explore",
            params=params,
            headers={"Authorization": f"Bearer {token}"}
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Strava API error")

    segs = resp.json().get("segments", [])[:3]
    result = []
    for s in segs:
        kom_time = s.get("kom", "")
        result.append({
            "id": s["id"],
            "name": s["name"],
            "distance": round(s["distance"] / 1000, 1),
            "avg_grade": s.get("avg_grade", 0),
            "kom": kom_time,
            "entry_count": s.get("entry_count", 0),
            "url": f"https://www.strava.com/segments/{s['id']}"
        })

    return {"segments": result}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
