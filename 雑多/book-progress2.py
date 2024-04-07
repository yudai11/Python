from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///books.db'
db = SQLAlchemy(app)

class Book(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    author = db.Column(db.String(100), nullable=False)
    pages = db.Column(db.Integer, nullable=False)
    progress = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f"{self.title} by {self.author}: {self.progress}/{self.pages}"

@app.route('/')
def index():
    books = Book.query.all()
    return render_template('index.html', books=books)

@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        title = request.form['title']
        author = request.form['author']
        pages = int(request.form['pages'])
        progress = int(request.form['progress'])

        book = Book(title=title, author=author, pages=pages, progress=progress)
        db.session.add(book)
        db.session.commit()

        return redirect(url_for('index'))
    else:
        return render_template('add.html')

@app.route('/update/<int:id>', methods=['GET', 'POST'])
def update(id):
    book = Book.query.get_or_404(id)
    if request.method == 'POST':
        book.title = request.form['title']
        book.author = request.form['author']
        book.pages = int(request.form['pages'])
        book.progress = int(request.form['progress'])
        db.session.commit()

        return redirect(url_for('index'))
    else:
        return render_template('update.html', book=book)

@app.route('/delete/<int:id>', methods=['POST'])
def delete(id):
    book = Book.query.get_or_404(id)
    db.session.delete(book)
    db.session.commit()

    return redirect(url_for('index'))

if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)

