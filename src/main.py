import base64
import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import openai  # Make sure to configure your client

# read key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
openai.api_key = OPENAI_API_KEY

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def camera_home(request: Request):
    return templates.TemplateResponse("camera_photo_select/code.html", {"request": request})

@app.post("/preview", response_class=HTMLResponse)
async def preview_photo(request: Request, photo_data: str = Form(...)):
    return templates.TemplateResponse("fragments/preview.html", {"request": request, "photo_data": photo_data})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_photo(request: Request, photo_data: str = Form(...)):
    # Convert base64 → bytes for OpenAI API
    header, encoded = photo_data.split(",", 1)
    image_bytes = base64.b64decode(encoded)

    # Example: call an OpenAI model (GPT-4o-mini or vision endpoint)
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "Describe the food in this image."},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded}"}
            ]}
        ]
    )
    result_text = response.choices[0].message.content if response.choices else "No result"

    return templates.TemplateResponse("fragments/result.html", {"request": request, "result": result_text})
