
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/add', methods=['GET'])
def add():
    return jsonify({'result': 2 + 2})

if __name__ == '__main__':
    app.run(debug=True)
