from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template('main.html')


@app.route('/', methods=['POST'])
def my_form_post():
    """
        Takes the comment submitted by the user, does some stuff to it,
        then returns it and renders it in the my-form.html template
    """

    text = request.form['text']

    processed_text = text.upper()

    return render_template('main.html', text=text, processed_text=processed_text)


if __name__ == '__main__':
    app.run(debug=True)
