from flask import Flask, render_template, url_for, request, redirect
from collections import deque
from Transcript_Scrape import scrape_transcript
# Require html5lib package

app = Flask(__name__)

CHAT_LOG = deque()


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        content = request.form['User-Text'] # Get chat
        if content != '': 
            CHAT_LOG.append(content)
        FILE = request.files['FILE'] # Get file (if there)
        if FILE.filename != '':
            FILE.save('Transcript.xls') # Save file for preprocessing and analyzing
            scrape_transcript('Transcript.xls')
    return render_template('index.html', queue = CHAT_LOG)

if __name__ == "__main__":
    app.run(debug=True)