## Meraki Nexus Artwork Classification Backend </br>
This project is a FastAPI server that evaluates artwork using a pre-trained PyTorch (EfficientNet) model. It accepts an image URL, downloads the image, and returns a calculated "market value score" along with the top 5 predicted art style classifications.
<br>
## 🛠️ Tech Stack
* **Backend:** FastAPI, Uvicorn
* **Machine Learning:** PyTorch, TorchVision
* **Image Handling:** Pillow (PIL)
* **HTTP/Downloads:** `requests`, `gdown`
### Installation $ Development
</br>Run:
```bash
#Install/check your dependencies (you already have them, but torch and torchvision are key):
pip install -r requirements.txt
#Run The Server
fastapi dev main.py
```

The server will be available at http://127.0.0.1:8000. </br>
Live api: meraki-nexus-ai.onrender.com

---

## API Reference

**Base URL:** `/nexus-ai`

### Pradictions

#### Create a Prediction
**POST** `/evaluate`  

**Body**
```json
{
  "image_url": "https://cdn.shopify.com/s/files/1/2283/9155/files/Springtime.jpg?v=1649170021"
}

```

**Response 200**
```json
{
    "status": "success",
    "predictions": [
        {
            "class_name": "amateur",
            "confidence": 0.9509398937225342
        },
        {
            "class_name": "fauvism",
            "confidence": 0.025065984576940536
        },
        {
            "class_name": "chinese_landscape",
            "confidence": 0.014497017487883568
        },
        {
            "class_name": "art_nouveau",
            "confidence": 0.006360294297337532
        },
        {
            "class_name": "abstract_art",
            "confidence": 0.0015162513591349125
        }
    ]
}
```

---


