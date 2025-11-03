import io
import gdown
import requests
from PIL import Image
from fastapi import HTTPException

def download_image(url: str) -> Image.Image:
    try:
        output_bytes = gdown.download_bytes(url, quiet=True)
        if output_bytes:
            image = Image.open(io.BytesIO(output_bytes))
            return image
        else:
            raise ValueError("Not a GDrive link or gdown failed.")
            
    except Exception:
        try:
            response = requests.get(url)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            return image
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {e}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"URL did not contain a valid image. Error: {e}")