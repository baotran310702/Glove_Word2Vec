import gensim.downloader
from flask import Flask, jsonify
from flask_cors import CORS
import argparse

parser = argparse.ArgumentParser(description='Run the Gensim GloVe backend server.')
parser.add_argument('--host', type=str, default='127.0.0.1', help='Host IP address to bind to')
args = parser.parse_args()


# Tạo một ứng dụng Flask
app = Flask(__name__)

# Sử dụng Flask-CORS
CORS(app)

glove_model = gensim.models.KeyedVectors.load("glove-wiki-gigaword-300.model")

# Function to recommend similar words
def recommend_similar_words(word, topn=15):
    try:
        similar_words = glove_model.most_similar(word, topn=15)
     
        return similar_words
    except KeyError:
        return f"'{word}' is not in the vocabulary."

# Định nghĩa một route khác
@app.route('/api/recommend-similar-words/<word>')
def greet(word):
    try:
        similar_words = glove_model.most_similar(word, topn=15)
        # Chỉ lấy ra top 5 từ tương tự nhất
        top_similar_words = [word[0] for word in similar_words[:15]]
        return jsonify({'similar_words': top_similar_words})
    except KeyError:
        return jsonify({'error': f"'{word}' is not in the vocabulary."})
    
# Định nghĩa một route khác
@app.route('/')
def helloWorld():
    return jsonify({"message": "Hello, World!"})

# Chạy ứng dụng trên localhost với cổng 5000
if __name__ == '__main__':
    app.run(host=args.host, port=5000, debug=True)

