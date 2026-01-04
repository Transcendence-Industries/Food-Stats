from fastapi import FastAPI, File, UploadFile, HTTPException

from pipeline import MainPipeline, TotalReport

app = FastAPI()
pipeline = MainPipeline()


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> TotalReport:
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    try:
        file_bytes = await file.read()
        return pipeline.run(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline Exception: {e}")
