import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser
import streamlit as st

# Web Crawling
def crawl(url, depth, max_depth=5):
    if depth > max_depth:
        return []
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]
        data = [soup.get_text()]
        for link in links:
            data.extend(crawl(link, depth + 1, max_depth))
        return data
    except:
        return []

# Data Chunking and Embedding
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()

def chunk_data(data):
    chunks = []
    for text in data:
        chunks.extend(text.split('\n\n'))
    return chunks

data = crawl('https://docs.nvidia.com/cuda/', 1)
chunks = chunk_data(data)
embeddings = [embed_text(chunk) for chunk in chunks]

# Milvus Vector Database
connections.connect("default", host="localhost", port="19530")

fields = [
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="metadata", dtype=DataType.STRING)
]
schema = CollectionSchema(fields, "doc_embeddings")
collection = Collection("documents", schema)
metadata = [f"chunk_{i}" for i in range(len(chunks))]

collection.insert([embeddings, metadata])

# Hybrid Retrieval
schema = Schema(content=TEXT(stored=True))
ix = create_in("indexdir", schema)
writer = ix.writer()

for chunk in chunks:
    writer.add_document(content=chunk)
writer.commit()

def search_bm25(query):
    with ix.searcher() as searcher:
        query = QueryParser("content", ix.schema).parse(query)
        results = searcher.search(query)
        return [r['content'] for r in results]

def retrieve_with_dpr(query):
    inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True)
    query_embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
    # Implement the retrieval logic with Milvus here, using the query_embedding
    return []

def re_rank(results):
    # Implement re-ranking logic based on relevance and similarity
    return results

# Question Answering
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def answer_question(question, context):
    return qa_pipeline(question=question, context=context)

# Streamlit Interface
st.title("NLP Question Answering System")
query = st.text_input("Enter your query:")
if query:
    bm25_results = search_bm25(query)
    dpr_results = retrieve_with_dpr(query)
    combined_results = bm25_results + dpr_results
    re_ranked_results = re_rank(combined_results)
    answers = [answer_question(query, context) for context in re_ranked_results]
    st.write(answers)
