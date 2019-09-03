from flask import Flask, render_template, url_for, flash, redirect
from services.forms import KeywordForm 
from services.analysis import Analysis
import json
import os

app = Flask(__name__)


with open(os.path.dirname(os.path.abspath(__file__))+'/files/config.json') as config_file:
    data = json.load(config_file)
    app.config['SECRET_KEY'] = data['SECRET_KEY']

analysis = Analysis()

@app.route('/', methods=['GET', 'POST'])
def home():
    form = KeywordForm()
    if form.validate_on_submit():
        print('Recieve Request for: '+form.first_keyword.data+' '+form.second_keyword.data)
        result = analysis.analyse_tweets(form.first_keyword.data, form.second_keyword.data)
        if result.get('no_tweets'):
            flash('There are no tweets')
        else:
            return render_template('results.html', result=result, trend=form.first_keyword.data, personality=form.second_keyword.data)
    return render_template('home.html', form=form, show_loader=False)

#to start using python filename.py
if __name__ == '__main__':
    app.run(debug=True)
    # app.run(debug=True, host="0.0.0.0", port=6130, use_reloader=False, threaded=True)

