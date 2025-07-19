# Modal Anki 

Modal Anki is a tool to generate flashcards from PDFs and save them to Anki.

We use a reasoning model to generate the flashcards, based on the text in the PDF. Then, we save the flashcards to an importable Anki package. Have fun!

Curently training runs on a 2 A100 GPUs with a Qwen/QwQ-32B-AWQ model. Sign up for an account at [Modal](https://modal.com) to run the training. You can get 30$/month of free credits.

## Usage

Basic usage:
```bash
modal run flashcards.py::pdf_flashcards --pdf-path="path/to/your.pdf"
```

To specify the number of cards per chunk:
```bash
modal run flashcards.py::pdf_flashcards --pdf-path="path/to/your.pdf" --num-cards=15
```

