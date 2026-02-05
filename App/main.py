import io
import uuid
from PIL import Image
from fastapi import FastAPI, Request
from fastapi import UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse

from Inference.colorize import colorize
from Inference.model import get_model

model = get_model()

version = "v1"

app = FastAPI(
    version=version
)

app.mount(
    "/static",
    StaticFiles(directory="App/static/"),
    name="static"
)

templates = Jinja2Templates(directory="App/templates/")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "Image Denoising Demo"
        }
    )


@app.post("/process")
def process(image: UploadFile = File(...)):
    job_id = str(uuid.uuid4())

    contents = image.file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    input_path = f"App/static/inputs/{job_id}_input.png"
    output_path = f"App/static/outputs/{job_id}_output.png"

    img.save(input_path)

    img_colorized = colorize(img, model, save_path=output_path)

    return RedirectResponse(
        url=f"/result/{job_id}",
        status_code=303
    )


@app.get("/result/{job_id}", response_class=HTMLResponse)
def show_result(request: Request, job_id: str):
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "input_image": f"/static/inputs/{job_id}_input.png",
            "output_image": f"/static/outputs/{job_id}_output.png"
        }
    )
