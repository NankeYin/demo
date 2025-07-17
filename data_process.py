import json
import os
import pandas as pd
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama

# Configuration paths
train_path = "E:\\demo\\demo\\data\\train.csv"
test_path = "E:\\demo\\demo\\data\\test.csv"
output_dir = "E:\\demo\\demo\\processed_data"
os.makedirs(output_dir, exist_ok=True)
# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB)")


def load_and_preprocess(train_path, test_path):
    """Load and preprocess Banking77 dataset"""
    # Load CSV files
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Merge datasets
    full_df = pd.concat([train_df, test_df])

    # Standardize column names
    required_columns = {'text', 'category'}
    if required_columns.issubset(full_df.columns):
        # Original Banking77 format
        full_df = full_df.rename(columns={'text': 'question', 'label': 'category_id'})
    elif 'question' in full_df.columns and 'category' in full_df.columns:
        # Custom format
        full_df = full_df.rename(columns={'category': 'category_name'})
    else:
        missing = required_columns - set(full_df.columns)
        raise ValueError(f"CSV missing required columns: {', '.join(missing)}")

    # Create category mapping
    if 'category_name' not in full_df.columns:
        categories = full_df['category_id'].unique()
        category_mapping = {idx: name for idx, name in enumerate(categories)}
        full_df['category_name'] = full_df['category_id'].map(category_mapping)

    # Save processed data
    full_df.to_csv(os.path.join(output_dir, "full_dataset.csv"), index=False)
    with open(os.path.join(output_dir, "category_mapping.json"), 'w') as f:
        json.dump(category_mapping, f)

    return full_df


def generate_definitions_locally(df):
    """Generate term definitions using local Ollama"""
    # Group questions by category
    grouped = df.groupby('category_name')['question'].apply(list).reset_index()

    definitions = []
    for _, row in grouped.iterrows():
        category = row['category_name']
        examples = "\n".join(row['question'][:3])  # Take first 3 examples

        # Generate definition using local Ollama
        response = ollama.generate(
            model='qwen:7b',
            system="You are a banking expert. Create concise definitions (max 30 words) for financial terms.",
            prompt=f"Term: {category}\nExample questions:\n{examples}\nDefinition:"
        )

        # Extract just the definition text
        definition = response['response'].strip()
        if "Definition:" in definition:
            definition = definition.split("Definition:")[-1].strip()

        definitions.append(definition)

    # Create terms dataframe
    terms_df = grouped[['category_name']].copy()
    terms_df.columns = ['term']
    terms_df['definition'] = definitions
    terms_df['examples'] = grouped['question'].apply(lambda x: " | ".join(x[:3]))

    return terms_df


def create_vector_store(terms_df):
    """Create FAISS vector store with local embeddings"""
    # Prepare documents
    documents = []
    for _, row in terms_df.iterrows():
        content = f"""
        Term: {row['term']}
        Definition: {row['definition']}
        Examples: {row['examples']}
        """
        documents.append(Document(page_content=content))

    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "]
    )
    splits = text_splitter.split_documents(documents)

    # Initialize embedding model with GPU support
    model_kwargs = {'device': device}
    encode_kwargs = {
        'batch_size': 32,
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
    vectorstore.save_local(os.path.join(output_dir, "faiss_index"))
    print(f"Vector store saved to {output_dir}/faiss_index")

    return vectorstore


def main():
    print("=" * 50)
    print("Banking77 Knowledge Base Pipeline")
    print("=" * 50)

    try:
        # Step 1: Load data
        print("\n[1/3] Loading and preprocessing data...")
        full_df = load_and_preprocess(train_path, test_path)
        print(f"Loaded {len(full_df)} records with {full_df['category_name'].nunique()} categories")

        # Step 2: Generate definitions
        print("\n[2/3] Generating term definitions...")
        terms_df = generate_definitions_locally(full_df)
        terms_df.to_csv(os.path.join(output_dir, "bank_terms.csv"), index=False)
        print(f"Generated definitions for {len(terms_df)} terms")

        # Step 3: Create vector store
        print("\n[3/3] Creating vector store...")
        vectorstore = create_vector_store(terms_df)
        print(f"Created vector store with {vectorstore.index.ntotal} vectors")

        print("\n" + "=" * 50)
        print("Processing complete! Files saved to:", output_dir)
        print("=" * 50)

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()