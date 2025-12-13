from flask import Flask, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'FeatureBot v1.0',
        'timestamp': pd.Timestamp.now().isoformat()
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Welcome to FeatureBot',
        'endpoints': {'/health': 'Check if model is running'},
        'github': 'https://github.com/Abeeba540/FeatureBot_EDA_Project_on_Adult_Income_Dataset'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
