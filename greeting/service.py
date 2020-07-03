from flask import request,render_template
from flask import jsonify
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('hello.html')

@app.route('/hello',methods=['POST'])
def hello():
    message = request.get_json(force=True)
    name = message['name']
    response ={
        'greeting': 'hello, '+name+'!'
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)    