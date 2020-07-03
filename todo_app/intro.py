"""first you need to install virutal environment
>pip3 install virtualenv
virtual environement allows you to work with others easily 

Next step is creating the env itself 
>virtualenv env
Note that env is the name of the environment

activate the environement 
>env\Scripts\activate

https://www.youtube.com/watch?v=X7mg3pJfhZQ

"""

from flask import Flask

app=Flask(__name__)

@app.route('/')

def index():
    return "hello world!"

if __name__ == "__main__":
    app.run(debug=True)

