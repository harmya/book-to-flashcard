import os
import subprocess
import json
from typing import Any, List, Dict, Tuple
import modal

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.9.1",
        "huggingface_hub[hf_transfer]==0.32.0",
        "flashinfer-python==0.2.6.post1",
        "aiohttp",
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
MODEL_DIR = "/model"
MODEL_NAME = "Qwen/QwQ-32B-AWQ"
VLLM_PORT = 8000


@app.cls(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    timeout=60 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
class FlashcardGenerator:
    @modal.enter()
    def start_server(self):
        """Start the VLLM server when the container starts"""

        cmd = [
            "vllm",
            "serve",
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

    def create_flashcard_prompt(self, text: str, num_cards: int = 20) -> str:
        return f"""
        
        You are an expert educator creating quiz-style flashcards that test deep understanding and practical application.
        CRITICAL: You must respond with ONLY a valid JSON object. Do not include any other text, explanations, or commentary.
        FLASHCARD OBJECTIVES:
        - Test conceptual understanding, not just definitions
        - Challenge learners to apply concepts to new situations  
        - Require reasoning and problem-solving
        - Help identify knowledge gaps through active recall
        - Connect concepts across different contexts
    
        QUESTION TYPES TO USE:
        1. Scenario-based: "Given situation X, what would happen if...?"
        2. Cause-effect: "Why does X lead to Y? What are the underlying mechanisms?"
        3. Comparison: "How would you distinguish between X and Y in practice?"
        4. Application: "How would you use concept X to solve problem Y?"
        5. Troubleshooting: "If you observed symptom X, what could be the causes?"
        6. Design decisions: "When would you choose approach X over Y?"
        7. Prediction: "What would be the consequences of changing X?"
        8. Analysis: "What factors contribute to outcome X?"
    
        AVOID:
        - Yes/no questions
        - Purely factual recall without reasoning
        - Questions answered by memorizing lists
        - Acronym explanations unless testing understanding of the concept
    
        QUESTION STEMS TO PREFER:
        - "Why would you..."
        - "How could you determine..."
        - "What factors would influence..."
        - "In what scenarios would..."
        - "How would you troubleshoot..."
        - "What would be the trade-offs of..."
        - "How would changing X affect..."
        - "What evidence would suggest..."
    
        Create exactly {num_cards} flashcards from the following text.
        DO NOT MAKE FLASHCARDS IF THE TEXT IS NOT RELEVANT TO THE SUBJECT (acknowledgements, references, table of contents, etc.).
    
        IMPORTANT: Respond with ONLY this JSON structure:
    
        {{
          "flashcards": [
            {{
              "front": "Scenario-based question that requires reasoning and application",
              "back": "Answer that explains the reasoning process and key principles involved"
            }}
          ]
        }}
    
        TEXT TO PROCESS:
        {text[:10000]}
    
        Focus on creating questions that make learners think critically and apply concepts rather than simply recall facts."""

    def pdf_pages_generator(self, file_path):
        import pymupdf

        doc = pymupdf.open(file_path)

        pages = []
        try:
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                pages.append(text)

            return pages

        finally:
            doc.close()

    def extract_text_chunks(
        self, pdf_bytes: bytes, chunk_size: int = 32
    ) -> List[Tuple[int, str]]:
        path = "/tmp/flashcards_temp.pdf"
        with open(path, "wb") as f:
            f.write(pdf_bytes)

        chunks = []
        current_text = []
        chunk_index = 0
        page_count = 0

        try:
            for page_text in self.pdf_pages_generator(path):
                current_text.append(page_text)
                page_count += 1

                if page_count % chunk_size == 0:
                    full_text = "\n\n".join(current_text)
                    chunks.append((chunk_index, full_text))
                    current_text = []
                    chunk_index += 1

            if current_text:
                full_text = "\n\n".join(current_text)
                chunks.append((chunk_index, full_text))

            return chunks
        finally:
            if os.path.exists(path):
                os.remove(path)

    async def send_request_to_model(self, prompt: str) -> str:
        import aiohttp
        import asyncio

        messages = [
            {
                "role": "system",
                "content": "You are an expert educator who creates exceptional learning materials. Always respond with valid JSON when requested.",
            },
            {"role": "user", "content": prompt},
        ]

        payload = {
            "messages": messages,
            "model": MODEL_NAME,
            "stream": False,
            "temperature": 0.3,
            "top_p": 0.5,
            "max_tokens": 4096,
        }

        headers = {"Content-Type": "application/json"}

        max_retries = 30

        async with aiohttp.ClientSession() as session:
            for attempt in range(max_retries):
                try:
                    async with session.post(
                        f"http://localhost:{VLLM_PORT}/v1/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=600),
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]

                except aiohttp.ClientConnectorError:
                    if attempt < max_retries - 1:
                        print(
                            f"Server not ready, retrying in 10 seconds... (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(10)
                    else:
                        raise Exception(
                            "VLLM server failed to start within expected time"
                        )
                except Exception as e:
                    raise Exception(f"Request failed: {e}")

    def extract_and_clean_json(self, response_text: str) -> dict:
        import re

        text = response_text.strip()
    
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        json_start = -1
        json_end = -1
        brace_count = 0
        
        for i, char in enumerate(text):
            if char == '{':
                if json_start == -1:
                    json_start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and json_start != -1:
                    json_end = i + 1
                    break
        
        if json_start == -1 or json_end == -1:
            raise ValueError("No valid JSON object found in response")
        
        json_text = text[json_start:json_end]
        
        json_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_text)
        json_text = re.sub(r',\s*}', '}', json_text)
        json_text = re.sub(r',\s*]', ']', json_text)
        
        return json.loads(json_text)

    async def process_text_chunk(
        self, chunk_data: Tuple[int, str], num_cards: int = 16
    ) -> Dict[str, Any]:
        chunk_index, text = chunk_data

        try:
            print(f"Processing chunk {chunk_index}...")
            prompt = self.create_flashcard_prompt(text, num_cards)
            response = await self.send_request_to_model(prompt)

            flashcards_data = self.extract_and_clean_json(response)

            if "flashcards" not in flashcards_data:
                raise ValueError("Response doesn't contain 'flashcards' key")

            valid_flashcards = []
            for i, card in enumerate(flashcards_data["flashcards"]):
                if "front" not in card or "back" not in card:
                    print(
                        f"Chunk {chunk_index}: Flashcard {i} missing 'front' or 'back' field"
                    )
                    continue
                valid_flashcards.append(card)

            print(
                f"Generated {len(valid_flashcards)} valid flashcards for chunk {chunk_index}"
            )

            return {
                "success": True,
                "chunk_index": chunk_index,
                "flashcards": valid_flashcards,
                "text_length": len(text),
            }

        except json.JSONDecodeError as e:
            print(f"Chunk {chunk_index}: JSON parsing error: {e}")
            return {
                "success": False,
                "chunk_index": chunk_index,
                "error": f"Failed to parse AI response as JSON: {e}",
                "raw_response": response[:1000]
                if "response" in locals()
                else "No response",
            }
        except Exception as e:
            print(f"Chunk {chunk_index}: Error processing chunk: {e}")
            return {"success": False, "chunk_index": chunk_index, "error": str(e)}

    @modal.method()
    async def generate_flashcards_from_pdf(
        self, pdf_bytes: bytes, num_cards: int = 16, chunk_size: int = 32
    ) -> Dict[str, Any]:
        import asyncio

        try:
            print("Extracting text chunks from PDF...")
            text_chunks = self.extract_text_chunks(pdf_bytes, chunk_size)

            if not text_chunks:
                return {"success": False, "error": "No text chunks extracted from PDF"}

            print(f"Extracted {len(text_chunks)} chunks of {chunk_size} pages each")

            print("Processing all chunks concurrently...")

            chunk_tasks = []
            for chunk_index, text in text_chunks:
                chunk_data = (chunk_index, text)
                task = self.process_text_chunk(chunk_data, num_cards)
                chunk_tasks.append(task)

            print(f"Starting {len(chunk_tasks)} concurrent chunk processing tasks...")

            results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

            all_flashcards = []
            successful_chunks = 0
            failed_chunks = 0
            total_text_length = 0

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_chunks += 1
                    print(f"Chunk {i} failed with exception: {result}")
                elif result.get("success", False):
                    all_flashcards.extend(result["flashcards"])
                    successful_chunks += 1
                    total_text_length += result.get("text_length", 0)
                    chunk_index = result.get("chunk_index", i)
                    print(
                        f"Chunk {chunk_index} completed: {len(result['flashcards'])} flashcards"
                    )
                else:
                    failed_chunks += 1
                    chunk_index = (
                        result.get("chunk_index", i) if isinstance(result, dict) else i
                    )
                    error = (
                        result.get("error", "unknown error")
                        if isinstance(result, dict)
                        else "unknown error"
                    )
                    print(f"Chunk {chunk_index} failed: {error}")

            print("\nConcurrent processing complete!")
            print(f"Successful chunks: {successful_chunks}")
            print(f"Failed chunks: {failed_chunks}")
            print(f"Total flashcards: {len(all_flashcards)}")

            return {
                "success": True,
                "flashcards": all_flashcards,
                "metadata": {
                    "num_cards_generated": len(all_flashcards),
                    "total_chunks": len(text_chunks),
                    "successful_chunks": successful_chunks,
                    "failed_chunks": failed_chunks,
                    "total_text_length": total_text_length,
                    "chunk_size": chunk_size,
                },
            }

        except Exception as e:
            print(f"Error in generate_flashcards_from_pdf: {e}")
            return {"success": False, "error": str(e)}


@app.local_entrypoint()
def pdf_flashcards(pdf: str = None, num_cards: int = 32, chunk_size: int = 16):
    if pdf is None:
        print("Please provide a PDF path using --pdf argument")
        return

    if not os.path.exists(pdf):
        print(f"PDF file not found: {pdf}")
        return

    with open(pdf, "rb") as f:
        pdf_bytes = f.read()

    generator = FlashcardGenerator()

    print(f"Processing PDF: {pdf}")
    print(f"Chunk size: {chunk_size} pages")
    print(f"Cards per chunk: {num_cards}")

    result = generator.generate_flashcards_from_pdf.remote(
        pdf_bytes, num_cards, chunk_size
    )

    if result["success"]:
        metadata = result["metadata"]
        print(f"\nSuccessfully generated {len(result['flashcards'])} flashcards!")
        print("Processing stats:")
        print(f"   - Total chunks: {metadata['total_chunks']}")
        print(f"   - Successful: {metadata['successful_chunks']}")
        print(f"   - Failed: {metadata['failed_chunks']}")
        print(f"   - Chunk size: {metadata['chunk_size']} pages")
        print(f"   - Total text length: {metadata['total_text_length']} chars")

        print("\nSample flashcards:")
        for i, card in enumerate(result["flashcards"][:3]):
            print(f"\n--- Sample Flashcard {i + 1} ---")
            print(f"Q: {card['front']}")
            print(f"A: {card['back']}")

        # Save to JSON file
        output_file = pdf.replace(".pdf", "_flashcards.json")
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nFlashcards saved to: {output_file}")

    else:
        print(f"Error generating flashcards: {result['error']}")
        if "raw_response" in result:
            print(f"Raw AI response: {result['raw_response']}")

    # Convert to Anki
    from make_cards import convert_json_file_to_anki
    convert_json_file_to_anki(output_file, pdf.replace(".pdf", "_flashcards.apkg"))
    print(f"Anki file saved to: {pdf.replace('.pdf', '_flashcards.apkg')}")
