from flask import Flask
app = Flask(__name__)

@app.route('/hello')
def hello_world():
    return 'Hello World!'

@app.route('/')
def index_page():
    return "index page"

if __name__ == '__main__':
    app.debug = True
    app.run()