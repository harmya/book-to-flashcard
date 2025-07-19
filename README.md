# Book to Flashcard 

I would like have certain concepts stay fresh in my mind. I think flashcards are a good way to keep your knowledge "warm" so it does not get evicted. However, I struggle to find the time to make these flashcards. So I wrote this tool to generate flashcards from PDFs and save them to Anki.

We use a reasoning model to generate the flashcards, based on the text in the PDF. Then, we save the flashcards to an importable Anki package. Have fun!

Curently training runs on a 2 A100 GPUs on Modal with a Qwen/QwQ-32B-AWQ model. Sign up for an account at [Modal](https://modal.com) to run the training. You can get 30$/month of free credits.

## Usage

Basic usage:
```bash
modal run flashcards.py::pdf_flashcards --pdf-path="path/to/your.pdf"
```

To specify the number of cards per chunk:
```bash
modal run flashcards.py::pdf_flashcards --pdf-path="path/to/your.pdf" --num-cards=15
```

## Example
There should be an Anki pkg in the data/ folder that makes flashcards from a distributed systems book. Feel free to try any book that you want!



