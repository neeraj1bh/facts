# Facts Trivia Program

## Overview

This program is designed to engage users with trivia facts, enabling inquiries into a collection of trivia facts stored within a `facts.txt` file. It utilizes the `langchain-openai` and other related packages to process and respond to user queries with relevant trivia information.

## Requirements

- Python 3.11
- Key Dependencies: langchain, openai, python-dotenv, tiktoken, chromadb, matplotlib, numpy, langchain-openai
- See `requirements.txt` for a complete list.

## Project Structure

- `Pipfile` and `Pipfile.lock` for managing Python package dependencies.
- `facts.txt`: Contains the trivia facts.
- `main.py`: The main script to run the trivia program.
- `prompt.py`, `redundant_filter_retriever.py`, `scores.py`: Auxiliary modules.
- `emb/`: Directory containing embedding data and `chroma.sqlite3` database.
- `scores.ipynb`: Jupyter notebook for analyzing scores.

## Setup

1. Install dependencies using Pipenv: `pipenv install`.
2. Ensure all required environment variables are set (check `.env` template if provided).

## Usage

Execute the program to start interacting with the trivia fact system:

```bash
python main.py
```
