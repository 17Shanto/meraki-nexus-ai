import torch
from torchvision import transforms

ART_STYLE_WEIGHTS = {
    "Abstract Expressionism": 100,
    "Pop Art": 95,
    "Baroque": 90,
    "Renaissance": 85,
    "Cubism": 80,
    "Surrealism": 75,
    "Art Nouveau": 70,
    "Minimalism": 65,
    "Expressionism": 60,
    "Futurism": 55,
    "High Renaissance": 50,
    "Romanticism": 45,
    "Realism": 40,
    "Symbolism": 35,
    "Constructivism": 30,
    "Op Art": 25,
    "Art Deco": 20,
    "Chinese Landscape": 15,
    "Post-Impressionism": 10,
    "Abstract Art": 0,
    "Neo-Classicism": 5
}

# Create lists from the dictionary, preserving order
CLASS_NAMES = list(ART_STYLE_WEIGHTS.keys())
WEIGHTS_LIST = list(ART_STYLE_WEIGHTS.values())


preprocess = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model(model_path: str):
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}")
        print("       The server cannot start.")
        exit(1)
    except Exception as e:
        print(f"ERROR loading scripted model: {e}")
        print("      This model file might be corrupt or not a TorchScript model.")
        exit(1)

    print("Model loaded successfully.")
    
    weights_tensor = torch.tensor(WEIGHTS_LIST, dtype=torch.float32).to(device)
    total_weight = weights_tensor.sum()
    
    return model, device, weights_tensor, total_weight


MODEL_PATH = "./artwork-model/artwork_classification_model_efficientNet.pth"
model, device, weights_tensor, total_weight = load_model(MODEL_PATH)