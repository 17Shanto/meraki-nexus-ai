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
    "market_value_score": 0.347290575504303,
    "predictions": [
        {
            "class_name": "Renaissance",
            "confidence": 0.9453256130218506
        },
        {
            "class_name": "Romanticism",
            "confidence": 0.9269545674324036
        },
        {
            "class_name": "Baroque",
            "confidence": 0.9255679249763489
        },
        {
            "class_name": "Op Art",
            "confidence": 0.8090653419494629
        },
        {
            "class_name": "Post-Impressionism",
            "confidence": 0.4038437306880951
        }
    ]
}
```

---


