from fastapi import FastAPI, HTTPException
from PIL import Image
import torch
import torch.nn.functional as F # Import functional

# Import components from our other files
from schemas import ImageRequest
from utils import download_image
from model import (
    model, device, preprocess, 
    CLASS_NAMES # Simplified imports
)


app = FastAPI(
    title="Image Evaluation API",
    description="API to evaluate an image and calculate its market value score."
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Evaluation API. Post to /nexus-ai/evaluate/"}


@app.post("/nexus-ai/evaluate/")
async def evaluate_image(request: ImageRequest):
    try:
        image = download_image(request.image_url)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device) 

        with torch.no_grad(): 
            output = model(input_batch)

        # CHANGED: Use softmax (dim=1) as in your new eval script
        probabilities = F.softmax(output, dim=1)[0]


        if len(probabilities) != len(CLASS_NAMES):
            raise HTTPException(
                status_code=500, 
                detail=f"Model output mismatch. Expected {len(CLASS_NAMES)} classes, got {len(probabilities)}"
            )

        
        # REMOVED: weighted_sum and market_value calculation
        
        top5_probs, top5_indices = torch.topk(probabilities, 5)
        predictions = [
            {
                "class_name": CLASS_NAMES[idx.item()], 
                "confidence": prob.item()
            }
            for prob, idx in zip(top5_probs, top5_indices)
        ]

        # UPDATED: Return value no longer includes market_value_score
        return {
            "status": "success",
            "predictions": predictions
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")