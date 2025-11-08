import torch
from torchvision import transforms

# NEW CLASS LIST
# I have cleaned the list you provided (removed duplicates, fixed spelling).
# YOU MUST VERIFY this is the correct order from your training data.
CLASS_NAMES = [
    "abstract_art",
    "abstract_expressionism",
    "amateur",
    "art_nouveau",
    "baroque",
    "chinese_landscape",
    "constructivism",
    "cubism",
    "expressionism",
    "fauvism",
    "futurism",
    "high_renaissance",
    "minimalism",
    "op_art",
    "pop_art",
    "post_impressionism",
    "realism",
    "renaissance", # Corrected spelling from "reniassance"
    "romanticism",
    "surrealism",
    "symbolism"
]


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
    
    # Removed weights_tensor and total_weight
    return model, device


MODEL_PATH = "./artwork-model/artwork_classification_model_subject_2_efficientNet.pth"
# Simplified the return values
model, device = load_model(MODEL_PATH)