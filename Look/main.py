from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse,FileResponse
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from model_gan import upscaling_function
import os 

app=FastAPI()

template=Jinja2Templates(directory="templates")
app.mount("/static",StaticFiles(directory="static"),name="static")

STATIC="static/image"
IMAGE_PATH = "static/image/unclear_image.jpeg"
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

for file in os.listdir(STATIC):
    os.remove(os.path.join(STATIC,file))


@app.get("/",response_class=HTMLResponse)
async def home(request:Request): 
    return template.TemplateResponse("index.html",{"request":request})



@app.post("/upload")
async def send(request:Request,file:UploadFile=File(...)):
    if not file.content_type.startswith("image/"):
        return HTTPException(status_code=400,detail="invalid input")

    file_name="unclear_image.jpeg"
    up_file_name="super_resolution.jpeg"
    
    os.makedirs("static",exist_ok=True)
    save_path=f"static/image/{file_name}"
    with open(save_path,"wb") as f:
        f.write(await file.read())
    
  

    # go=True if file_name else False 
    # ,"cap":go

    upscaling_function(IMAGE_PATH,SAVED_MODEL_PATH)
    
    upscaled=f"/static/image/{up_file_name}"
    return template.TemplateResponse("index.html",{"request":request,"capture":save_path,"super":upscaled})

IMG_PATH="static/image/super_resolution.jpeg"
@app.get("/download")
async def download_image(request:Request):
    if os.path.exists(IMG_PATH):
        return FileResponse(
            path=IMG_PATH,
            filename="super_resolution.jpeg",  # name for the downloaded image
            media_type="image/jpeg"  # MIME type for JPEG images
        )
        
    

    else:
        return template.TemplateResponse("index.html",{"request":request})