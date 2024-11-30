from flask import Flask, jsonify, request
from flask_cors import CORS
import base64
from digit_recog_model import inference_function
from utils import prepare_image

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# Enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

# For testing
predictions = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def base64_to_bytes(base64_str, output_file_path = None):
    """Converts base64 string into a bytes object and saves it on disk"""
    header = 'data:image/png;base64,'

    if base64_str.startswith(header):
        base64_str = base64_str[len(header):]
    print("String with removed header: ", base64_str)
    
    img_bytes = base64.b64decode(base64_str)

    with open(output_file_path, 'wb') as f:
        f.write(img_bytes)


# New route
@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify('pong!')

@app.route('/predict', methods=['POST'])
def predict():
    # Get request and save image data
    request_data = request.get_json()
    image_bs64 = request_data['image']
    # print("Got image in base64 format: ", image_bs64)
    # print('base64 image type: ', type(image_bs64))
    base64_to_bytes(image_bs64, 'img_buffer_dir/digit.png')
    img_tensor = prepare_image('img_buffer_dir')
    ix, conf_score = inference_function(img_tensor)
    

    # Start constructing the response obj
    response_obj = {'status': 'success'}
    response_obj['message'] = 'Received your request data!'
    # Return a prediction
    response_obj['prediction'] = predictions[0]
    return jsonify(response_obj)

if __name__ == '__main__':
    app.run()