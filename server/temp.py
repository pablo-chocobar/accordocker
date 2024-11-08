from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from actualutils import parse_and_process_image, main_workflow , generate_speech
import warnings
from chat_memory import ChatMemory

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app)

chat_memory = ChatMemory()
locallama = "llama3.1"

from ollama import Client
client = Client(host='http://ollama:11434')

@app.route('/ask', methods=['POST'])
def submit():
    image_desc = None

    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                image = parse_and_process_image(file)
                image_desc = main_workflow(image)
            except Exception as e:
                print("Error getting image description:", str(e))
                return jsonify({'error': str(e)}), 400
        else:
            return jsonify({'error': 'Invalid file type'}), 400

    question = request.form.get('text')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    history = chat_memory.get_history()
    history_context = "\n".join([f"User: {conv['user']}\nSystem: {conv['system']}" for conv in history])
    print("Current history context:", history_context)  # Debug print

    if image_desc:
        chat_memory.add_conversation("describe what's in the image", image_desc)
        # Update history context with the new image description
        history = chat_memory.get_history()
        history_context = "\n".join([f"User: {conv['user']}\nSystem: {conv['system']}" for conv in history])

    system_prompt = f'''
You are an AI assistant describing scenes to a visually impaired person based on image analysis. Your responses should be concise yet thorough, covering all key details provided in the image description. Avoid unnecessary elaboration or assumptions beyond what's explicitly stated. Use clear, descriptive language that helps create a mental image of the scene. Do not use phrases like "You are standing" or assume the person's position. Instead, describe the scene objectively.

Conversation so far:
{history_context}

{f"Current image description: {image_desc}" if image_desc else ""}

Provide a single paragraph response (roughly 3-5 sentences) that covers the main elements of the scene, including:
1. Key features of the environment (buildings, shops, stalls)
2. Notable objects or people
3. Weather conditions or other relevant atmospheric details
Start with a specific detail from the scene rather than a general overview.'''

    response = client.chat(model=locallama, messages=[
        {
            'role': 'system',
            'content': system_prompt
        },
        {
            'role': 'user',
            'content': f'Hey Shreeram, {question}'
        }
    ])
    
    ans = response['message']['content']
    print("Model response:", ans)  # Debug print

    chat_memory.add_conversation(question, ans)

    tts_response = generate_speech(ans)
    return send_file(tts_response, as_attachment=False)

@app.route('/follow-up', methods=['POST'])
def follow_up():
    follow_up_question = request.form.get('text')

    if not follow_up_question:
        return jsonify({'error': 'No follow-up question provided'}), 400

    history = chat_memory.get_history()
    history_context = "\n".join([f"User: {conv['user']}\nSystem: {conv['system']}" for conv in history])
    print("Follow-up history context:", history_context)  # Debug print

    response = client.chat(model=locallama, messages=[
        {
            'role': 'system',
            'content': f'''
You are an AI assistant continuing the conversation with a visually impaired person. Here is the conversation so far:

{history_context}

Answer the user's next question using the context and information already provided. Be pleasant, and do not mention that the user is blind or anything'''
        },
        {
            'role': 'user',
            'content': f'Hey Shreeram, {follow_up_question}'
        }
    ])
    
    follow_up_ans = response['message']['content']
    print("Follow-up response:", follow_up_ans)  # Debug print

    chat_memory.add_conversation(follow_up_question, follow_up_ans)

    tts_response = generate_speech(follow_up_ans)

    return send_file(tts_response, as_attachment=False)

@app.route('/clear-history', methods=['POST'])
def clear_history():
    chat_memory.clear_history()
    return jsonify({'message': 'Conversation history cleared'}), 200

if __name__ == "__main__":
    app.run()