from argparse import Namespace
from flask import Flask,request
from flask_restx import Resource,Api

app = Flask(__name__)


@app.route("/test",methods=["POST"])
def test():
    
    data = request.get_json(silent=True, cache=False, force=True)

    print("Received data:", data["title"],data["content"])

    return {
        "title" : data["title"][0],
        "content" : data["content"][0]
    }



if __name__ == "__main__":
    app.run(debug=True,host="127.0.0.1",port = 5000)