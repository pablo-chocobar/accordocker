import torch
from PIL import Image
import concurrent.futures
import ollama
from transformers import AutoProcessor, AutoModelForCausalLM
import warnings, requests
warnings.filterwarnings("ignore")

# Initialize conversation history
conversation_history = []

def initialize_models():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    
    return model, processor, device, torch_dtype

model, processor, device, torch_dtype = initialize_models()
locallama = "llama3.1"

def parse_and_process_image(file):
    image = Image.open(file.stream)
    wpercent = (256 / float(image.size[0]))
    hsize = int((float(image.size[1]) * float(wpercent)))
    image = image.resize((256, hsize), Image.Resampling.LANCZOS)
    return image

def classify_task(user_input):
    response = ollama.chat(model=locallama, messages=[
      {
        'role': 'system',
        'content': '''You are Shreeram, you will help me classify the further prompts into one of these three tasks. 1) Image captioning, 2) OCR. 
        You will output only the number corresponding to the task and nothing else.'''
        },
        { 
            'role': 'user',
            'content': 'Hey Shreeram, ' + user_input,
        }
    ])
    return response['message']['content']

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

def parallel_image_tasks(image, model, processor, device, torch_dtype):
    prompt1 = "<OCR>"
    prompt2 = "<MORE_DETAILED_CAPTION>"

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(image_task, prompt1, image, model, processor, device, torch_dtype)
        future2 = executor.submit(image_task, prompt2, image, model, processor, device, torch_dtype)

        ans1 = future1.result()
        ans2 = future2.result()

    return ans1, ans2

def main_workflow(image):
    global conversation_history

    # task = classify_task(question)
    task = "1"
    if task == "1":
        prompt = "<MORE_DETAILED_CAPTION>"
        result = image_task(prompt, image, model, processor, device, torch_dtype)
        return result

    return None

def ask_follow_up(question):
    global conversation_history

    # Combine conversation history for context in follow-up
    history_context = "\n".join([f"User: {conv['user']}\nSystem: {conv['system']}" for conv in conversation_history])

    response = ollama.chat(model=locallama, messages=[
        {
            'role': 'system',
            'content': f'''
You are an AI assistant continuing the conversation with a visually impaired person. Here is the conversation so far:

{history_context}

Answer the user's next question using the context and information already provided.'''},
        {
            'role': 'user',
            'content': 'Hey Shreeram, ' + question
        }
    ])

    text_res = response['message']['content']
    print(text_res)

    # Append follow-up question and answer to the conversation history
    conversation_history.append({
        'user': question,
        'system': text_res
    })

    return text_res


def text_to_speech(text):
    url = "http://alltalk-tts:7851/api/tts-generate"
    payload = {
        "text_input": text,
        "text_filtering": "standard",
        "character_voice_gen": "female_01.wav",
        "narrator_enabled": "false",
        "narrator_voice_gen": "male_01.wav",
        "text_not_inside": "character",
        "language": "en",
        "output_file_name": "myoutputfile",
        "output_file_timestamp": "true",
        "autoplay": "true",
        "autoplay_volume": "0.8"
    }

    response = requests.post(url, data=payload)

    if response.status_code == 200:
        print("TTS Request was successful.")
        return response.json()
    else:
        print(f"Failed to make TTS request, status code: {response.status_code}")
        print("Response content:", response.text)
        return None
