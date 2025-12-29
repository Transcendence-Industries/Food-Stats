# Food Stats

Estimates meal nutrition from a photo by combining image segmentation, food classification, and a small nutrition lookup table behind a FastAPI service, with an optional n8n workflow that connects the API to a Telegram bot.

This is a portfolio/demo project. The output is an estimate, not a medical or dietary measurement.

## Highlights

- Detects food regions in an uploaded image with a Detectron2 Mask R-CNN segmentation model.
- Classifies each detected crop with an EfficientNet-B0 food classifier.
- Estimates grams and calories from the segmented area, serving-size assumptions, and calories per 100 g.
- Exposes the pipeline through a Dockerized FastAPI API.
- Includes an importable n8n workflow for a Telegram bot interface.
- Ships sample model artifacts so the API can be tried without retraining first.

## Demo

| Input image | Segmentation output |
| --- | --- |
| ![Input image](./screenshots/sample_input.jpg) | ![Segmented image](./screenshots/sample_segmentation.jpg) |

![Telegram chat using the n8n workflow](./screenshots/telegram_chat.jpg)

The bundled screenshots and model artifacts are from development models trained briefly for demonstration. Better results require longer training and stronger nutrition data.

## How It Works

1. A client uploads a meal photo to `POST /analyze`.
2. The segmentation model finds food masks and bounding boxes.
3. Each crop is classified into one of the supported food labels.
4. The nutrition database provides calories per 100 g and a default serving size for the predicted label.
5. The API returns per-food estimates plus total grams and calories.

The current portion estimate is intentionally simple: it uses each mask's image-area ratio and the food's configured serving size. This keeps the project easy to inspect, but it is also the main accuracy limitation.

## Tech Stack

- Python 3.9
- FastAPI and Uvicorn
- PyTorch, Torchvision, Detectron2, and OpenCV
- Pydantic response models
- Docker Compose
- n8n and Telegram integration
- `uv` for dependency management

## Repository Structure

```text
src/
  api.py                         FastAPI routes
  main.py                        API entry point
  pipeline.py                    End-to-end inference pipeline
  segmentation.py                Detectron2 segmentation wrapper
  classification.py              EfficientNet classifier wrapper
  database.py                    Nutrition lookup wrapper
  dev/
    segmentation/                Dataset preparation, training, inference scripts
    classification/              Dataset preparation, training, inference scripts
bin/
  segmentation/model_final.pth   Segmentation model artifact
  classification/model_final.pth Classification model artifact
  classification/mapping.json    Class index mapping
  database/db.json               Nutrition values and serving sizes
n8n/
  food-stats-telegram-workflow.json
screenshots/
```

## API Usage

Docker is the recommended way to run the project, because the locked Detectron2 and PyTorch wheels target Linux x86_64.

1. Copy the sample environment file if you do not already have a local one:

   ```bash
   cp stack.env.sample stack.env
   ```

2. Start the API:

   ```bash
   docker compose up -d --build
   ```

3. Open the API docs:

   ```text
   http://localhost:8080/docs
   ```

4. Analyze an image:

   ```bash
   curl -X POST "http://localhost:8080/analyze" \
     -F "file=@data/samples/sample_1.jpg"
   ```

Example response shape:

```json
{
  "foods": [
    {
      "label": "pizza",
      "grams": 180.4,
      "calories": 480.2
    }
  ],
  "total_grams": 180.4,
  "total_calories": 480.2
}
```

Health check:

```text
GET /health
```

## n8n and Telegram

Import `n8n/food-stats-telegram-workflow.json` into n8n to use Food Stats through Telegram.

The workflow:

- receives Telegram photo messages,
- downloads the highest-quality image,
- sends the image to `http://food-stats:8080/analyze`,
- replies with detected foods, grams, and calories.

The default API URL assumes n8n can resolve the Docker Compose service name `food-stats`. If n8n runs outside the same Docker network, update the HTTP Request node URL to the reachable API address, for example `http://localhost:8080/analyze`.

Telegram credentials are not included. Configure your own bot credentials after importing the workflow.

## Training

The development scripts support retraining the segmentation and classification models.

Datasets used:

- [FoodSeg103](https://huggingface.co/datasets/EduardoPacheco/FoodSeg103) for food segmentation
- [Food-101](https://huggingface.co/datasets/ethz/food101) for food classification
- [Open Food Facts product database](https://huggingface.co/datasets/openfoodfacts/product-database) for nutrition data

Training flow:

```bash
uv sync
uv run src/dev/segmentation/prepare_dataset.py
uv run src/dev/classification/prepare_dataset.py
uv run src/dev/segmentation/training.py
uv run src/dev/classification/training.py
uv run src/dev/segmentation/inference.py
uv run src/dev/classification/inference.py
```

After training, copy the model artifacts into the runtime paths:

```text
data/segmentation/model/model_final.pth -> bin/segmentation/model_final.pth
data/classification/model/model_final.pth -> bin/classification/model_final.pth
```

## Current Limitations

- Portion size is estimated from image area, not from depth, camera calibration, plate size, or user-provided scale.
- Nutrition values come from a compact JSON lookup table rather than a full normalized nutrition database.
- The sample models are development artifacts and are included to demonstrate the full pipeline.
- The locked local environment is optimized for Linux x86_64 Docker deployment. macOS ARM users should prefer Docker or adjust the ML dependencies.
