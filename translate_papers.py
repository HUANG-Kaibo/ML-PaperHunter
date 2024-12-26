import os
import json
import time
from typing import Dict, Tuple
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import argparse
from tqdm import tqdm

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_gpt4(client: OpenAI, system_prompt: str, user_prompt: str) -> Tuple[str, str]:
    """Call GPT-4-mini API with retry mechanism"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower randomness
            timeout=60  # Set timeout
        )
        
        content = response.choices[0].message.content
        lines = content.strip().split('\n')
        if len(lines) != 2:
            raise ValueError(f"Translation format error: expected 2 lines, got {len(lines)}")
        return lines[0].strip(), lines[1].strip()
    
    except Exception as e:
        print(f"API call error: {str(e)}")
        raise

def translate_paper(client: OpenAI, title: str, abstract: str) -> Tuple[str, str]:
    """Translate a single paper's title and abstract"""
    system_prompt = """You are a professional academic translator specialized in translating computer science papers from English to Chinese. 
Please translate the given title and abstract accurately and professionally.
Return exactly TWO lines:
Line 1: Chinese translation of the title
Line 2: Chinese translation of the abstract"""

    user_prompt = f"""Please translate this paper title and abstract to Chinese:

Title: {title}
Abstract: {abstract}

Return exactly two lines: translated title and translated abstract."""

    return call_gpt4(client, system_prompt, user_prompt)

def translate_papers(
    input_file: str = "data/relevant_papers.jsonl",
    output_file: str = "data/translated_papers.jsonl"
) -> None:
    """
    Read papers from input file, translate titles and abstracts, and save results.
    Supports checkpoint resume and incremental writing.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found!")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read all papers
    papers = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    paper = json.loads(line)
                    if not isinstance(paper, dict) or 'title' not in paper or 'abstract' not in paper:
                        print(f"Warning: Invalid data format at line {line_num}, skipping")
                        continue
                    papers.append(paper)
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON parse error at line {line_num}, skipping: {e}")
                    continue

    if not papers:
        raise ValueError(f"No valid paper data found in {input_file}")

    print(f"Successfully loaded {len(papers)} papers")

    # Get processed papers
    processed_titles = set()
    processed_title = None
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        current_json = ""
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                current_json += line
                if line.strip() == '}':
                    try:
                        paper = json.loads(current_json)
                        processed_title = paper.get('title')
                        processed_titles.add(processed_title)
                        current_json = ""
                    except json.JSONDecodeError:
                        pass

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_ENDPOINT")
    )

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    if not os.getenv("OPENAI_API_ENDPOINT"):
        raise ValueError("Please set OPENAI_API_ENDPOINT environment variable")

    # Find last processed position
    start_index = 0
    if processed_title:
        for i, paper in enumerate(papers):
            if paper['title'] == processed_title:
                start_index = i
                break
    
    # Process papers with tqdm progress bar
    try:
        with tqdm(total=len(papers)-start_index, desc="Translating papers") as pbar:
            for i in range(start_index, len(papers)):
                paper = papers[i]
                title = paper['title']
                
                if title in processed_titles:
                    pbar.update(1)
                    continue
                
                try:
                    title_zh, abstract_zh = translate_paper(
                        client, 
                        paper['title'],
                        paper['abstract']
                    )
                    
                    translated_paper = {
                        'title': title,
                        'title_zh': title_zh,
                        'abstract': paper['abstract'],
                        'abstract_zh': abstract_zh
                    }
                    
                    with open(output_file, "a", encoding="utf-8") as out_f:
                        json_str = json.dumps(translated_paper, ensure_ascii=False, indent=2)
                        out_f.write(json_str + "\n")
                    
                    processed_titles.add(title)
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"\nTranslation failed ({title}): {str(e)}")
                    translated_paper = {
                        'title': title,
                        'title_zh': f"[Translation Failed] {title}",
                        'abstract': paper['abstract'],
                        'abstract_zh': f"[Translation Failed] {paper['abstract']}"
                    }
                    with open(output_file, "a", encoding="utf-8") as out_f:
                        json_str = json.dumps(translated_paper, ensure_ascii=False, indent=2)
                        out_f.write(json_str + "\n")
                
                pbar.update(1)
                
    except Exception as e:
        print(f"\nProgress saved. Current progress: {i}/{len(papers)}")
        raise Exception(f"Error processing file: {str(e)}")

    print(f"\nTask completed! Translated papers saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate papers to Chinese')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file path')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file path')
    args = parser.parse_args()
    
    translate_papers(
        input_file=args.input,
        output_file=args.output
    ) 
