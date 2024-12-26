import os
import json
import time
from typing import List, Dict, Tuple
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import argparse
from tqdm import tqdm

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_gpt4(client: OpenAI, prompt: str) -> List[int]:
    """Call GPT-4 API with retry mechanism and process the response"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.choices[0].message.content
        
        # Parse returned index list
        indices = []
        for line in content.split('\n'):
            line = line.strip()
            if line and line != "None found.":
                try:
                    num = int(line.lstrip('- *#').strip())
                    indices.append(num)
                except ValueError:
                    continue
        
        return indices
    
    except Exception as e:
        print(f"API call error: {str(e)}")
        raise

def find_llm_security_papers(
    input_file: str = "data/accepted_papers.jsonl",
    output_file: str = "data/relevant_papers.jsonl",
    chunk_size: int = 10
) -> None:
    """
    Read data in chunks from input_file, use GPT-4 to filter papers related to LLM security
    based on titles and abstracts, and write results to output_file.
    Supports checkpoint resume and incremental writing.
    """
    # 1. Ensure data directories exist
    os.makedirs(os.path.dirname(input_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist!")

    # 2. Initialize environment checks
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    if not os.getenv("OPENAI_API_ENDPOINT"):
        raise ValueError("Please set OPENAI_API_ENDPOINT environment variable")

    # 3. Read papers from file
    papers = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    paper = json.loads(line)
                    if not isinstance(paper, dict) or 'title' not in paper or 'abstract' not in paper:
                        continue
                    papers.append(paper)
                except json.JSONDecodeError:
                    continue

    if not papers:
        raise ValueError(f"No valid paper data read from {input_file}")

    # Import prompt
    from prompts.llm_security import PROMPT

    # Get the last processed paper title
    last_processed_title = None
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        last_paper = json.loads(line)
                        last_processed_title = last_paper.get('title')
                    except json.JSONDecodeError:
                        pass

    # Find the last processing position
    start_chunk = 0
    if last_processed_title:
        for i, paper in enumerate(papers):
            if paper['title'] == last_processed_title:
                start_chunk = ((i + 1) // chunk_size) * chunk_size
                break

    # Process papers with tqdm progress bar
    with tqdm(total=len(papers), initial=start_chunk, desc="Processing papers") as pbar:
        for i in range(start_chunk, len(papers), chunk_size):
            chunk = papers[i:i + chunk_size]
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),base_url=os.getenv("OPENAI_API_ENDPOINT"))
            try:
                papers_text = "\n\n".join([
                    f"{idx+1}.\nTitle: {paper['title']}\nAbstract: {paper['abstract']}"
                    for idx, paper in enumerate(chunk)
                ])
                prompt = PROMPT.format(papers=papers_text)
                relevant_indices = call_gpt4(client, prompt)
                
                if relevant_indices:
                    with open(output_file, "a", encoding="utf-8") as f:
                        for idx in relevant_indices:
                            if 1 <= idx <= len(chunk):
                                paper = chunk[idx-1]
                                paper_data = {
                                    'title': paper['title'],
                                    'abstract': paper['abstract']
                                }
                                f.write(json.dumps(paper_data, ensure_ascii=False) + "\n")
                
                time.sleep(1)
                pbar.update(len(chunk))
                
            except Exception as e:
                print(f"Error processing chunk {i//chunk_size + 1}: {str(e)}")
                print("Progress saved, you can resume later")
                raise

    print(f"Task completed! LLM security related papers saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find LLM security related papers')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file path')
    args = parser.parse_args()
    
    find_llm_security_papers(
        input_file=args.input,
        output_file=args.output
    )
