import io
import os
import threading
import time
import uuid

import cv2
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "template"))

STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/upload/")
async def upload_image(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return templates.TemplateResponse("form.html", {
            "request": request,
            "error": "Invalid image file."
        })

    height, width = img.shape[:2]
    dummy_img = np.zeros(img.shape, np.uint8)

    img = cv2.resize(img, (0, 0), fx=1, fy=0.5)
    blur = cv2.GaussianBlur(img, (15, 15), 0)
    edges = cv2.Canny(img, 100, 200, 5)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    dummy_img[:height // 2, :] = blur
    dummy_img[height // 2:, :] = edges

    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(STATIC_DIR, filename)
    cv2.imwrite(filepath, dummy_img)

    return RedirectResponse(url=f"/result?filename={filename}", status_code=303)


@app.get("/result", response_class=HTMLResponse)
async def show_result(request: Request, filename: str):
    return templates.TemplateResponse("result.html", {
        "request": request,
        "image_url": f"/static/{filename}"
    })


@app.get("/result", response_class=HTMLResponse)
async def show_result(request: Request, filename: str):
    image_url = f"/static/{filename}"
    filepath = os.path.join(STATIC_DIR, filename)

    def delayed_delete(path):
        time.sleep(5)
        if os.path.exists(path):
            os.remove(path)

    threading.Thread(target=delayed_delete, args=(filepath,)).start()

    return templates.TemplateResponse("result.html", {
        "request": request,
        "image_url": image_url
    })

