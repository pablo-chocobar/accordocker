from flask import Flask, jsonify, request
from flask_cors import CORS
from utils import parse_and_process_image, main_workflow, ask_follow_up, text_to_speech
import warnings
import requests

warnings.filterwarnings("ignore")
OUTPUT_DIR = '/home/accord/alltalk_tts/outputs/'

app = Flask(__name__)
CORS(app)
conversation_history = []  # Global variable to store conversation history

@app.route('/ask', methods=['POST'])
def submit():
    global conversation_history
    
    image_message = None
    image = None
    
    
    if 'file' in request.files:
        file = request.files['file']
        print(file, file.filename)
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                url = 'http://192.168.0.129:5000/describe'

                file_content = file.read()
                files = {'image': (file.filename, file_content, file.content_type)}

                response = requests.post(url, files=files)

                response_data = response.json()
                print(response_data)
                image_desc = response_data.get("final response", "No response found")
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'Invalid file type'}), 400

    # Get the user's question from the form data
    question = request.form.get('text')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    ans = main_workflow(question, image_desc)

    tts_response = text_to_speech(ans)
    print(tts_response)

    # Store conversation history (both question and response)
    conversation_history.append({
        'user': question,
        'system': ans
    })

    # Return the response along with any image or question processing messages
    return jsonify({
        'image_message': image_message,
        'question_message': f"You asked: {question}",
        "final_response": ans,
        "audio": tts_response["output_file_url"]
    }), 200

@app.route('/follow-up', methods=['POST'])
def follow_up():
    global conversation_history

    # Get the follow-up question from the form data
    follow_up_question = request.form.get('text')
    if not follow_up_question:
        return jsonify({'error': 'No follow-up question provided'}), 400

    # Call the follow-up function
    follow_up_ans = ask_follow_up(follow_up_question)

    tts_response = text_to_speech(follow_up_ans)
    print(tts_response)

    # Return the follow-up response
    return jsonify({
        'follow_up_question': f"You asked: {follow_up_question}",
        'final_response': follow_up_ans,
        "audio": tts_response["output_file_url"]
    }), 200

@app.route("/")
def hello():
    return "<p>Hello, World!</p>"

if __name__ == "__main__":
    app.run()
