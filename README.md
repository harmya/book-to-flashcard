# Book to Flashcard 

**Convert any Book to ANKI flashcards**

I would like have certain concepts stay fresh in my mind. I read that flashcards are a good way to keep your knowledge "warm" so it does not get evicted. However, I struggle to find the time to make these flashcards. So I wrote this tool to generate flashcards from PDFs and save them to Anki.

We use a reasoning model to generate the flashcards, based on the text in the PDF. Then, we save the flashcards to an importable Anki package. Have fun!

Curently, inference runs on a 1 A100 GPUs on Modal with a Qwen/QwQ-32B-AWQ model. Sign up for an account at [Modal](https://modal.com) to run this. You can get 30$/month of free credits.

You can modify the prompt and and swap out the model to better *serve* your needs.

## Usage

Start by cloning this repo:
```bash
git clone https://github.com/harmya/book-to-flashcard.git
```

Basic usage:
```bash
modal run src/flashcards.py --pdf="path/to/your.pdf"
```

To specify the number of cards per pages-chunk:
```bash
modal run src/flashcards.py --pdf="path/to/your.pdf" --num-cards=16
```

To specify the number of cards per pages-chunk and the size of the chunk:
```bash
modal run src/flashcards.py --pdf="path/to/your.pdf" --num-cards=16 --chunk_size=32
```
This will break up the pdf into 32 page chunks and generate 16 cards per chunk.

## Example
There should be an Anki pkg in the data/ folder that makes flashcards from a distributed systems book. Feel free to try any book that you want!



