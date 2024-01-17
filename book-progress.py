import json

def add_book(books):
    title = input("Enter the book title: ")
    author = input("Enter the author's name: ")
    pages = int(input("Enter the total number of pages: "))
    progress = int(input("Enter the current page number: "))

    book = {"title": title, "author": author, "pages": pages, "progress": progress}
    books.append(book)

    with open("books.json", "w") as f:
        json.dump(books, f)

    print(f"{title} by {author} added to your reading list.")

def show_books(books):
    if not books:
        print("Your reading list is empty.")
    else:
        for i, book in enumerate(books):
            print(f"{i+1}. {book['title']} by {book['author']}: {book['progress']}/{book['pages']}")

def update_progress(books):
    show_books(books)
    index = int(input("Enter the number of the book you want to update: ")) - 1
    book = books[index]
    progress = int(input(f"Enter your progress in {book['title']}: "))
    book['progress'] = progress

    with open("books.json", "w") as f:
        json.dump(books, f)

    print(f"Your progress in {book['title']} has been updated to {progress}.")

def main():
    with open("books.json", "r") as f:
        books = json.load(f)

    while True:
        print("\nReading List Manager\n")
        print("1. Add book")
        print("2. Show books")
        print("3. Update progress")
        print("4. Quit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            add_book(books)
        elif choice == "2":
            show_books(books)
        elif choice == "3":
            update_progress(books)
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
