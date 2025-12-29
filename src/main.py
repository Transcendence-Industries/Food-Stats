import os
import logging
import uvicorn

import api

DEBUG = os.getenv("DEBUG", "False").lower() in {"1", "true", "yes", "on"}
API_PORT = int(os.getenv("API_PORT", "8080"))


def main_entrypoint():
    logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)

    logging.info(f"Starting server on port {API_PORT}...")
    uvicorn.run(api.app, host="0.0.0.0", port=API_PORT)


if __name__ == "__main__":
    main_entrypoint()
