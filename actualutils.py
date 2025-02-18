import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from TTS.api import TTS # Coqui TTS
import warnings
warnings.filterwarnings("ignore")

def initialize_models():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tts = TTS(model_name= 'tts_models/multilingual/multi-dataset/xtts_v2').to(device)

    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    
    return model, processor, device, torch_dtype ,tts

model, processor, device, torch_dtype, tts = initialize_models()
locallama = "llama3.1"

def parse_and_process_image(file):
    image = Image.open(file.stream)
    wpercent = (256 / float(image.size[0]))
    hsize = int((float(image.size[1]) * float(wpercent)))
    image = image.resize((256, hsize), Image.Resampling.LANCZOS)
    return image

def image_task(prompt, image, model, processor, device, torch_dtype):
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
    print(parsed_answer)

    return parsed_answer

def main_workflow(image, task = "1"):

    if task == "1":
        prompt = "<MORE_DETAILED_CAPTION>"
        result = image_task(prompt, image, model, processor, device, torch_dtype)
        return result
    elif task == "2":
        prompt = "<OCR>"
        result = image_task(prompt, image, model, processor, device, torch_dtype)
        return result

    return None

def generate_speech(text):
    tts.tts_to_file(text=text, file_path="./output.wav", language = "en", speaker = "Dionisio Schuyler")
    return "./output.wav"