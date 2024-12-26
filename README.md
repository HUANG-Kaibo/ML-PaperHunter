# ML-PaperHunter: A Tool for Topic-specific Paper Filtering from Top ML Conferences

This repository provides tools to track and analyze topic-specific papers from top ML conferences (ICML/ICLR/NeurIPS). It supports:

1. Automated paper crawling from OpenReview
2. Customizable topic-based paper filtering using LLMs
3. Machine translation of paper titles and abstracts

## Features

- **Automated Paper Crawling**: Fetch papers from OpenReview for ICML/ICLR/NeurIPS conferences
- **Flexible Topic Filtering**: Use LLM-based filtering with customizable topic definitions
- **Translation Support**: Translate filtered papers to other languages (currently supports Chinese)
- **Incremental Processing**: Support for resuming interrupted operations

## Data Processing Pipeline

1. **Paper Crawling**:
   - Uses OpenReview API to fetch paper data
   - Supports ICML 2024, ICLR 2024, and NeurIPS 2024
   - Crawling scripts: `get_icml24_data.ipynb`, `get_iclr24_data.ipynb`, `get_neurips24_data.ipynb`

2. **Topic-based Filtering**:
   - Uses GPT to identify papers related to specified topics
   - Customizable filtering criteria in `prompts/` directory
   - Processing script: `get_relatex_paper.py`

3. **Translation**:
   - Translates titles and abstracts to target language
   - Translation script: `translate_papers.py`

## Usage

### Prerequisites

1. Set up environment variables:
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_ENDPOINT="your-api-endpoint"
```

2. Install dependencies:
```bash
pip install openai tenacity tqdm jupyter
```

### Running the Pipeline

#### 1. Crawl Papers
Run the Jupyter notebooks to fetch papers from each conference:
```bash
jupyter notebook get_iclr24_data.ipynb
jupyter notebook get_icml24_data.ipynb 
jupyter notebook get_neurips24_data.ipynb
```

#### 2. Process Papers Using Shell Script
The `process_papers.sh` script automates the filtering and translation process:

```bash
# Make script executable
chmod +x process_papers.sh

# Run for a specific conference
./process_papers.sh data/iclr2024.jsonl

# Process multiple conferences
for conf in iclr2024 icml2024 neurips2024; do
    ./process_papers.sh data/${conf}.jsonl
done
```

The script will:
- Filter papers based on the specified topic
- Translate filtered papers
- Save results in the same directory with appropriate suffixes

#### 3. Customize Topic Filtering

1. Create a new prompt file in the `prompts` directory:
```python
# prompts/your_topic.py
PROMPT = """Review these numbered papers and list ONLY the numbers of papers related to [YOUR TOPIC].
[YOUR TOPIC] is defined as [YOUR DEFINITION].

If no papers are relevant, return "None found."

Papers to analyze:
{papers}"""
```

2. Update the import in `get_relatex_paper.py`:
```python
from prompts.your_topic import PROMPT
```

3. Run the processing pipeline with your custom topic:
```bash
./process_papers.sh data/iclr2024.jsonl
```

## Directory Structure

```
.
├── data/                    # Output directory for processed data
├── prompts/                 # Topic definition prompts
│   ├── llm_security.py     # Example topic: LLM security
│   └── your_topic.py       # Your custom topic
├── get_*_data.ipynb        # Conference-specific crawling notebooks
├── get_relatex_paper.py    # Topic filtering script
├── translate_papers.py     # Translation script
└── process_papers.sh       # Main processing script
```

## Acknowledgments

The paper crawling code is based on [OpenReview-paper-list](https://github.com/dw-dengwei/OpenReview-paper-list).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

