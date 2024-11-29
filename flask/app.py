from flask import Flask, jsonify, request
from flask_cors import CORS

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

print('Loading app')

# Enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

# New route
@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify('pong!')

@app.route('/predict', methods=['POST'])
def predict():
    response_obj = {'status': 'success'}
    request_data = request.get_json()
    print("Got this data from the front end: ", request_data)
    response_obj['message'] = 'Received your request data!'
    return jsonify(response_obj)

if __name__ == '__main__':
    app.run()