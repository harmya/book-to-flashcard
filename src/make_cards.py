import json
import genanki
import time
import os
from typing import Dict, Optional


class JSONToAnkiConverter:
    def __init__(self):
        self.model_id = int(time.time() * 1000000)
        self.deck_id = int(time.time() * 1000000) + 1

        self.model = genanki.Model(
            self.model_id,
            "Basic Flashcard Model",
            fields=[
                {"name": "Front"},
                {"name": "Back"},
            ],
            templates=[
                {
                    "name": "Card 1",
                    "qfmt": "{{Front}}",
                    "afmt": "{{FmtText:Back}}",
                },
            ],
            css="""
            .card {
                font-family: Arial, sans-serif;
                font-size: 16px;
                text-align: left;
                color: black;
                background-color: white;
                padding: 20px;
                line-height: 1.4;
            }
            
            .front {
                font-weight: bold;
                margin-bottom: 10px;
            }
            
            .back {
                margin-top: 10px;
            }
            
            code {
                background-color: #f5f5f5;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: monospace;
            }
            """,
        )

    def load_json_flashcards(self, file_path: str) -> Dict:
        """Load flashcards from JSON file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format - {e}")
            return {}

    def load_json_string(self, json_string: str) -> Dict:
        """Load flashcards from JSON string"""
        try:
            data = json.loads(json_string)
            return data
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format - {e}")
            return {}

    def create_deck(self, deck_name: str = "Generated Flashcards") -> genanki.Deck:
        return genanki.Deck(self.deck_id, deck_name)

    def convert_to_anki_cards(
        self, flashcard_data: Dict, deck_name: str = "Generated Flashcards"
    ) -> Optional[genanki.Deck]:
        if not flashcard_data.get("success", False):
            print("Error: Flashcard data indicates unsuccessful generation")
            return None

        flashcards = flashcard_data.get("flashcards", [])
        if not flashcards:
            print("Error: No flashcards found in data")
            return None

        deck = self.create_deck(deck_name)

        for i, card in enumerate(flashcards, 1):
            try:
                front = card.get("front", "").strip()
                back = card.get("back", "").strip()

                if not front or not back:
                    print(f"Warning: Skipping card {i} - missing front or back")
                    continue

                note = genanki.Note(model=self.model, fields=[front, back])

                deck.add_note(note)

            except Exception as e:
                print(f"Error processing card {i}: {e}")
                continue

        return deck

    def save_to_anki_package(
        self, deck: genanki.Deck, output_file: str = "flashcards.apkg"
    ):
        try:
            package = genanki.Package(deck)
            package.write_to_file(output_file)
            print(f"Successfully saved {len(deck.notes)} cards to '{output_file}'")
            return True
        except Exception as e:
            print(f"Error saving package: {e}")
            return False

    def convert_file_to_anki(
        self, json_file: str, output_file: str = None, deck_name: str = None
    ):
        flashcard_data = self.load_json_flashcards(json_file)
        if not flashcard_data:
            return False

        if not deck_name:
            deck_name = os.path.splitext(os.path.basename(json_file))[0]

        if not output_file:
            output_file = f"{deck_name}.apkg"

        deck = self.convert_to_anki_cards(flashcard_data, deck_name)
        if not deck:
            return False

        return self.save_to_anki_package(deck, output_file)

    def convert_string_to_anki(
        self,
        json_string: str,
        output_file: str = "flashcards.apkg",
        deck_name: str = "Generated Flashcards",
    ):
        flashcard_data = self.load_json_string(json_string)
        if not flashcard_data:
            return False

        deck = self.convert_to_anki_cards(flashcard_data, deck_name)
        if not deck:
            return False

        return self.save_to_anki_package(deck, output_file)

    def print_summary(self, flashcard_data: Dict):
        if flashcard_data.get("success"):
            metadata = flashcard_data.get("metadata", {})
            print("\n=== Flashcard Summary ===")
            print(f"Number of cards: {metadata.get('num_cards_generated', 'Unknown')}")
            print(
                f"Original text length: {metadata.get('text_length', 'Unknown')} characters"
            )
            print(
                f"Processed text length: {metadata.get('processed_text_length', 'Unknown')} characters"
            )

            cards = flashcard_data.get("flashcards", [])
            if cards:
                print("\nExample card:")
                print(f"Front: {cards[0]['front'][:100]}...")
                print(f"Back: {cards[0]['back'][:100]}...")


def convert_json_file_to_anki(json_file_path: str, output_name: str = None):
    converter = JSONToAnkiConverter()
    return converter.convert_file_to_anki(json_file_path, output_name)


def convert_json_string_to_anki(
    json_string: str,
    output_name: str = "flashcards.apkg",
    deck_name: str = "My Flashcards",
):
    converter = JSONToAnkiConverter()
    return converter.convert_string_to_anki(json_string, output_name, deck_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True)
    args = parser.parse_args()
    converter = JSONToAnkiConverter()
    converter.convert_file_to_anki(
        args.json, "flashcards.apkg", args.json.replace(".json", "")
    )
