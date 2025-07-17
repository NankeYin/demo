import json
import os
import pandas as pd
import requests
import time
import traceback

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration paths
train_path = "D:\\demo\\demo\\data\\train.csv"
test_path = "D:\\demo\\demo\\data\\test.csv"
output_dir = "D:\\demo\\demo\\processed_data"
os.makedirs(output_dir, exist_ok=True)

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
num_gpus = torch.cuda.device_count() if device == "cuda" else 0

print(f"Using device: {device}")
if device == "cuda":
    try:
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU VRAM: {vram:.2f} GB, Number of GPUs: {num_gpus}")
    except Exception as e:
        print(f"GPU detection failed: {str(e)}")


def load_and_preprocess(train_path, test_path):
    """Load and preprocess Banking77 dataset"""
    # Load CSV files
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    full_df = pd.concat([train_df, test_df])

    # Validate required columns
    required_columns = {'text', 'category'}
    if not required_columns.issubset(full_df.columns):
        missing = required_columns - set(full_df.columns)
        raise ValueError(f"CSV missing required columns: {', '.join(missing)}")

    # Standardize column names
    full_df = full_df.rename(columns={
        'text': 'question',
        'category': 'category_name'
    })

    # Create category mapping
    categories = full_df['category_name'].unique()
    category_mapping = {name: idx for idx, name in enumerate(categories)}
    full_df['category_id'] = full_df['category_name'].map(category_mapping)

    # Save processed data
    full_df.to_csv(os.path.join(output_dir, "full_dataset.csv"), index=False)
    with open(os.path.join(output_dir, "category_mapping.json"), 'w') as f:
        json.dump(category_mapping, f)

    return full_df


def generate_definitions_locally(df):
    """Generate term definitions using local Ollama"""
    grouped = df.groupby('category_name')['question'].apply(list).reset_index()
    definitions = []

    for _, row in grouped.iterrows():
        category = row['category_name']
        examples = "\n".join(row['question'][:3])
        definition = None
        retries = 3

        for attempt in range(retries):
            try:
                # Improved prompt for better responses
                prompt = (
                    f"Provide a concise technical definition for the banking term '{category}' "
                    f"based on these example questions:\n{examples}\n\n"
                    "Definition:"
                )

                payload = {
                    "model": "qwen:7b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_gpu": num_gpus,
                        "temperature": 0.7
                    }
                }

                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=45
                )
                response.raise_for_status()

                result = response.json()
                definition = result['response'].strip()

                # Handle different response formats
                if "Definition:" in definition:
                    definition = definition.split("Definition:", 1)[-1].strip()
                elif definition.startswith(f"Term: {category}"):
                    definition = definition.replace(f"Term: {category}\n", "", 1).strip()

                print(f"Generated definition for '{category}': {definition[:50]}...")
                break

            except requests.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}/{retries}): {str(e)}")
                time.sleep(2)
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                print(f"Processing error: {str(e)}")
                time.sleep(1)

        if definition is None:
            definition = f"Standard banking term related to {category}"
            print(f"Using fallback definition for '{category}'")

        definitions.append(definition)
        time.sleep(0.5)  # Rate limiting

    # Create terms dataframe
    terms_df = grouped[['category_name']].copy()
    terms_df.columns = ['term']
    terms_df['definition'] = definitions
    terms_df['examples'] = grouped['question'].apply(lambda x: " | ".join(x[:3]))

    return terms_df


def create_vector_store(terms_df):
    """Create FAISS vector store with local embeddings"""
    documents = []
    for _, row in terms_df.iterrows():
        content = (
            f"Term: {row['term']}\n"
            f"Definition: {row['definition']}\n"
            f"Examples: {row['examples']}"
        )
        documents.append(Document(page_content=content.strip()))

    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "]
    )
    splits = text_splitter.split_documents(documents)

    # Initialize embedding model
    model_kwargs = {'device': device}
    encode_kwargs = {
        'batch_size': min(32, len(splits)),
        'normalize_embeddings': True,
        'show_progress_bar': True
    }

    embedder = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embedder
    )

    # Save vector store
    save_path = os.path.join(output_dir, "faiss_index")
    vectorstore.save_local(save_path)
    print(f"Vector store saved to {save_path} with {vectorstore.index.ntotal} vectors")

    return vectorstore


def main():
    print("=" * 50)
    print("Banking77 Knowledge Base Pipeline")
    print("=" * 50)

    try:
        # Step 1: Load data
        print("\n[1/3] Loading and preprocessing data...")
        full_df = load_and_preprocess(train_path, test_path)
        print(f"Loaded {len(full_df)} records, {full_df['category_name'].nunique()} categories")

        # Step 2: Generate definitions
        if not os.path.exists(os.path.join(output_dir, "bank_terms.csv")):
            print("\n[2/3] Generating term definitions...")
            terms_df = generate_definitions_locally(full_df)
            terms_df.to_csv(os.path.join(output_dir, "bank_terms.csv"), index=False)
            print(f"Generated definitions for {len(terms_df)} terms")
        else:
            print("\n[2/3] Using existing term definitions...")
            terms_df = pd.read_csv(os.path.join(output_dir, "bank_terms.csv"))

        # Step 3: Create vector store
        print("\n[3/3] Creating vector store...")
        create_vector_store(terms_df)

        print("\n" + "=" * 50)
        print("Processing complete! Files saved to:", output_dir)
        print("=" * 50)

    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()