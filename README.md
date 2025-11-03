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


