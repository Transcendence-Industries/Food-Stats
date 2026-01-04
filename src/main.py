import os
import logging
import uvicorn

import api

DEBUG = bool(os.environ["DEBUG"])
API_PORT = int(os.environ["API_PORT"])


def main_entrypoint():
    logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)

    logging.info(f"Starting server on port {API_PORT}...")
    uvicorn.run(api.app, host="0.0.0.0", port=API_PORT)


if __name__ == "__main__":
    main_entrypoint()
