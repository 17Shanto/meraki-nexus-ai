import torch
from torchvision import transforms

ART_STYLE_WEIGHTS = {
    "abstract_expressionism": 1000.0,
    "surrealism": 917.4,
    "pop_art": 875.8,
    "abstract_art": 821.6,
    "post_impressionism": 550.2,
    "cubism": 461.9,
    "minimalism": 405.3,
    "expressionism": 367.9,
    "chinese_landscape": 343.9,
    "art_nouveau": 139.4,
    "constructivism": 148.9,
    "fauvism": 168.6,
    "op_art": 189.3,
    "realism": 120.7,
    "symbolism": 94.3,
    "futurism": 70.2,
    "romanticism": 56.5,
    "high_renaissance": 37.5,
    "renaissance": 27.7,
    "baroque": 23.6,
    "amateur": 12.8
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