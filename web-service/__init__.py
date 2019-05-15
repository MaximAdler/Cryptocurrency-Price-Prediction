import os
import subprocess

from flask import Flask
from flask import render_template, request
from flask_api import status



def create_app(test_config=None):

    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'web-service.sqlite'),
    )

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    def main():
        return render_template('base.html')

    @app.route('/monitoring')
    def monitoring():
        return render_template('monitoring/index.html')

    @app.route('/predicting')
    def predicting():
        return render_template('predicting/index.html')

    @app.route('/predict', methods=('POST',))
    def predict():
        if request.method == 'POST':
            response = subprocess.call(['sh', 'run_predictor.sh'])
            if response != 0:
                return ('SERVICE_UNAVAILABLE', status.HTTP_503_SERVICE_UNAVAILABLE, )
            return ('OK', status.HTTP_200_OK, )



    return app
