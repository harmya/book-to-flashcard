import os
import subprocess
from modal import Image, App, Secret, gpu, web_server, method
MODEL_DIR = "/model"
MODEL_NAME = "Qwen/QwQ-32B"
import json
from typing import Any, List, Dict
import modal
import PyPDF2
import io
import genanki

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.9.1",
        "huggingface_hub[hf_transfer]==0.32.0",
        "flashinfer-python==0.2.6.post1",
        "PyPDF2",
        "pdfplumber",
        "requests",
        "pymupdf",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .env({"VLLM_USE_V1": "1"})
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

FAST_BOOT = True
app = modal.App("PDF-Flashcards")
N_GPU = 1
MINUTES = 60
VLLM_PORT = 8000

@app.cls(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=15 * MINUTES,
    timeout=20 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
class FlashcardGenerator:
    @modal.enter()
    def start_server(self):
        """Start the VLLM server when the container starts"""
        import subprocess
        import time
        
        cmd = [
            "vllm",
            "serve",
            "--uvicorn-log-level=info",
            MODEL_NAME,
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
        ]
        cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
        cmd += ["--tensor-parallel-size", str(N_GPU)]
        
        print("Starting VLLM server with command:", " ".join(cmd))
        self.server_process = subprocess.Popen(" ".join(cmd), shell=True)
        
        time.sleep(60)


    def create_flashcard_prompt(self, text: str, num_cards: int = 20) -> str:
        """Create a detailed prompt for generating learning-oriented flashcards"""
        return f"""You are an expert educator and learning specialist. Your task is to create high-quality flashcards from the provided text that promote deep understanding and first-principles learning.

CRITICAL: You must respond with ONLY a valid JSON object. Do not include any other text, explanations, or commentary.

GUIDELINES FOR FLASHCARD CREATION:

1. **First Principles Focus**: Break down complex concepts into fundamental building blocks
2. **Active Recall**: Design questions that require retrieval, not recognition
3. **Conceptual Understanding**: Focus on "why" and "how" rather than just "what"
4. **Progressive Complexity**: Include both foundational and advanced concepts
5. **Practical Application**: Connect theory to real-world applications when possible

FLASHCARD DESIGN PRINCIPLES:
- Front: Clear, specific question or prompt
- Back: Comprehensive but concise answer with key insights
- Avoid yes/no questions
- Use scenarios and examples
- Include connections between concepts
- Emphasize cause-and-effect relationships

Create exactly {num_cards} flashcards from the following text. 

IMPORTANT: Respond with ONLY this JSON structure, nothing else:

{{
  "flashcards": [
    {{
      "front": "Question or prompt that tests deep understanding",
      "back": "Clear, comprehensive answer with key insights and connections"
    }}
  ]
}}

TEXT TO PROCESS:
{text[:8000]}

Remember: Focus on understanding, not memorization. Each flashcard should help build genuine comprehension of the subject matter. RESPOND WITH ONLY THE JSON OBJECT."""

    async def send_request_to_model(self, prompt: str) -> str:
        """Send request to the local VLLM server"""
        import requests
        import time
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert educator who creates exceptional learning materials. Always respond with valid JSON when requested."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        payload = {
            "messages": messages,
            "model": MODEL_NAME,
            "stream": False,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_tokens": 4000,
        }
        
        headers = {"Content-Type": "application/json"}
        
        # Wait for server to be ready
        max_retries = 30
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"http://localhost:{VLLM_PORT}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    print(f"Server not ready, retrying in 10 seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(10)
                else:
                    raise Exception("VLLM server failed to start within expected time")
            except Exception as e:
                raise Exception(f"Request failed: {e}")

    def pdf_pages_generator(self, file_path):
        """Generator that yields text page by page"""
        doc = fitz.open(file_path)
    
        try:
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                yield text
            
        finally:
            doc.close()

    @method()
    async def generate_flashcards_from_pdf(
        self, 
        pdf_bytes: bytes, 
        num_cards: int = 20,
    ) -> Dict[str, Any]:
        """
        Main method to process PDF and generate flashcards
        
        Args:
            pdf_bytes: Raw PDF file bytes
            num_cards: Number of flashcards to generate
            chunk_size: Size of text chunks for processing
            
        Returns:
            Dictionary containing flashcards and metadata
        """
        
        try:
            current_text = []
            curr_page = 0
            flashcards = []

            print("Extracting text from PDF...")
            for page_text in self.pdf_pages_generator(pdf_bytes):
                current_text.append(page_text)  
                curr_page += 1
                if curr_page % 32 == 0:
                    full_text = "\n\n".join(current_text)
                    prompt = self.create_flashcard_prompt(full_text, num_cards)
                    response = await self.send_request_to_model(prompt)
                    current_text = []
                    print(f"Generated {len(response)} flashcards for chunk {curr_page // 32}")
                    flashcards.append(response)
            
            all_flashcards = []

            for flashcard in flashcards:
                try:
                    flashcard_json = flashcard.strip()
                
                    json_start = flashcard_json.find('{')
                    json_end = flashcard_json.rfind('}') + 1
                
                    if json_start != -1 and json_end > json_start:
                        flashcard_json = flashcard_json[json_start:json_end]
                
                    if flashcard_json.startswith("```json"):
                        flashcard_json = flashcard_json[7:]
                    if flashcard_json.startswith("```"):
                        flashcard_json = flashcard_json[3:]
                    if flashcard_json.endswith("```"):
                        flashcard_json = flashcard_json[:-3]
                    flashcard_json = flashcard_json.strip()
                
                    flashcards_data = json.loads(flashcard_json)
                
                    if "flashcards" not in flashcards_data:
                     raise ValueError("Response doesn't contain 'flashcards' key")

                    for i, card in enumerate(flashcards_data["flashcards"]):
                        if "front" not in card or "back" not in card:
                            print(f"Flashcard {i} missing 'front' or 'back' field")
                            continue

                    all_flashcards.extend(flashcards_data["flashcards"])
                    
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}")
                    print(f"Raw response: {response[:500]}...")
                    return {
                        "success": False,
                        "error": f"Failed to parse AI response as JSON: {e}",
                        "raw_response": response[:1000]
                    }

                return {
                    "success": True,
                    "flashcards": all_flashcards,
                    "metadata": {
                        "num_cards_generated": len(all_flashcards),
                        "text_length": len(full_text),
                        "processed_text_length": len(full_text)
                    }
                }
                
            
            
        except Exception as e:
            print(f"Error in generate_flashcards_from_pdf: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Local entrypoint for testing
@app.local_entrypoint()
async def test_pdf_flashcards(pdf_path: str = None, num_cards: int = 10):
    """
    Test the PDF flashcard generator locally
    
    Usage:
        modal run flashcards.py::test_pdf_flashcards --pdf-path="path/to/your.pdf" --num-cards=15
    """

    if pdf_path is None:
        print("Please provide a PDF path using --pdf-path argument")
        return
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return  
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    generator = FlashcardGenerator()
    
    print(f"Processing PDF: {pdf_path}")
    result = generator.generate_flashcards_from_pdf.remote(pdf_bytes, num_cards)
    
    if result["success"]:
        print(f"\n✅ Successfully generated {len(result['flashcards'])} flashcards!")
        print(f"Metadata: {result['metadata']}")
        
        for i, card in enumerate(result["flashcards"][:3]):
            print(f"\n--- Flashcard {i+1} ---")
            print(f"Front: {card['front']}")
            print(f"Back: {card['back']}")
        
        # Save to JSON file
        output_file = pdf_path.replace(".pdf", "_flashcards.json")
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Flashcards saved to: {output_file}")
        
    else:
        print(f"❌ Error generating flashcards: {result['error']}")
        if "raw_response" in result:
            print(f"Raw AI response: {result['raw_response']}")