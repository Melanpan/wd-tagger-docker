import os
import fastapi
import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
import PIL
import logging
import time
from Utils import dbimutils

# Setup environment variables
MOAT_MODEL_REPO = os.environ.get("MOAT_MODEL_REPO", "SmilingWolf/wd-v1-4-moat-tagger-v2")
SWIN_MODEL_REPO = os.environ.get("SWIN_MODEL_REPO", "SmilingWolf/wd-v1-4-swinv2-tagger-v2")
CONV_MODEL_REPO = os.environ.get("CONV_MODEL_REPO", "SmilingWolf/wd-v1-4-convnext-tagger-v2")
CONV2_MODEL_REPO = os.environ.get("CONV2_MODEL_REPO", "SmilingWolf/wd-v1-4-convnextv2-tagger-v2")
VIT_MODEL_REPO = os.environ.get("VIT_MODEL_REPO", "SmilingWolf/wd-v1-4-vit-tagger-v2")
USE_CUDA = os.environ.get("USE_CUDA", 0)
MODEL = os.environ.get("MODEL", "SWIN")
CACHE_PATH = os.environ.get("CACHE_PATH", "/tmp/model_cache")
CHARACTER_THRESHOLD = float(os.environ.get("CHARACTER_THRESHOLD", "0.85"))
GENERAL_THRESHOLD = float(os.environ.get("GENERAL_THRESHOLD", "0.35"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class inferenceModel:
    model = None
    log = logging.getLogger(__name__)
    
    def __init__(self, model_repo: str, providers: list) -> None:
        self.model_repo = model_repo
        self.model_name = model_repo.split("/")[-1]
        self.providers = providers
    
    def load_model(self) -> None:
        self.log.info("Loading model")
        model_path = huggingface_hub.hf_hub_download(repo_id=self.model_repo, 
                                                     filename="model.onnx",
                                                     local_dir=CACHE_PATH)
        self.log.info(f"Model path: {model_path}, providers: {self.providers}")
        self.model = rt.InferenceSession(model_path, 
                                         providers=self.providers)
        self.log.info("Model loaded")
    

    def load_labels(self) -> None:
        self.log.info("Loading labels")
        label_path = huggingface_hub.hf_hub_download(MOAT_MODEL_REPO, 
                                                     "selected_tags.csv",
                                                     local_dir=CACHE_PATH)
        
        df = pd.read_csv(label_path)
        
        self.tag_names = df["name"].tolist()
        self.rating_indexes = list(np.where(df["category"] == 9)[0])
        self.general_indexes = list(np.where(df["category"] == 0)[0])
        self.character_indexes = list(np.where(df["category"] == 4)[0])

        self.log.info("Labels loaded")
    
    # https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags/blob/main/app.py
    def predict(self, image: PIL.Image.Image) -> dict:
        if self.model is None:
            raise ValueError("Model not loaded")
        
        rawimage = image
        _, height, width, _ = self.model.get_inputs()[0].shape

        # Alpha to white
        image = image.convert("RGBA")
        new_image = PIL.Image.new("RGBA", image.size, "WHITE")
        new_image.paste(image, mask=image)
        image = new_image.convert("RGB")
        image = np.asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = dbimutils.make_square(image, height)
        image = dbimutils.smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        probs = self.model.run([label_name], {input_name: image})[0]

        labels = list(zip(self.tag_names, probs[0].astype(float)))

        # First 4 labels are actually ratings: pick one with argmax
        ratings_names = [labels[i] for i in self.rating_indexes]
        rating = dict(ratings_names)

        # Original code filtered by threshold, but we want to return all tags
        # And leave the filtering to the client
        general_names = [labels[i] for i in self.general_indexes]        
        general_res = general_res = [x for x in general_names if x[1] > GENERAL_THRESHOLD]
        general_res = dict(general_res)

        character_names = [labels[i] for i in self.character_indexes]
        character_res = [x for x in character_names if x[1] > CHARACTER_THRESHOLD]
        character_res = dict(character_res)

        # sort by confidence
        general_res = {k: v for k, v in sorted(general_res.items(), key=lambda item: item[1], reverse=True)}
        character_res = {k: v for k, v in sorted(character_res.items(), key=lambda item: item[1], reverse=True)}

        return {"ratings": rating, "general": general_res, "characters": character_res}

def get_model_repo() -> str:
    if MODEL == "MOAT":
        return MOAT_MODEL_REPO
    elif MODEL == "SWIN":
        return SWIN_MODEL_REPO
    elif MODEL == "CONV":
        return CONV_MODEL_REPO
    elif MODEL == "CONV2":
        return CONV2_MODEL_REPO
    elif MODEL == "VIT":
        return VIT_MODEL_REPO
    else:
        raise ValueError("Unknown model")

model = inferenceModel(get_model_repo(), 
                       ["CUDAExecutionProvider"] if bool(USE_CUDA) else ["CPUExecutionProvider"])
model.load_model()
model.load_labels()
app = fastapi.FastAPI()

# simple html form for testing
@app.get("/predict")
def get_predict():
    html = """
    <body>
    <form action="/predict" enctype="multipart/form-data" method="post">
    <input name="image" type="file">
    <input type="submit">
    </form>
    </body>
    """
    return fastapi.responses.HTMLResponse(html)

@app.post("/predict")
def post_predict(image: fastapi.UploadFile = fastapi.File(...)):
    image = PIL.Image.open(image.file)
    logger.info(f"Predicting image ({image.size}) ")
    return model.predict(image)

@app.get("/")
def index():
    return {
        "model": MODEL, 
        "model_repo": model.model_repo, 
        "model_loaded": model.model is not None,
        "providers": model.model.get_providers(),
        "labels_loaded": model.tag_names is not None}
