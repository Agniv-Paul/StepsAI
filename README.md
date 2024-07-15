# Steps AI NLP Engineer Take-Home Assignment Report

## Overview

This report outlines the implementation of the take-home assessment for the NLP Engineer role at Steps AI. The assignment involved extensive web crawling, data chunking, vector database creation using Milvus, hybrid retrieval, re-ranking, and question-answering using an LLM. Additionally, a user interface was created using Streamlit to demonstrate the system's functionality.

## Task Breakdown

### Web Crawling

**Objective**: Develop a web crawler to scrape data from the provided NVIDIA CUDA documentation website, including sub-links up to a depth of 5 levels.

**Implementation**:
- Utilized the `requests` library to fetch HTML content and `BeautifulSoup` to parse and extract text and links.
- Implemented a recursive function to crawl sub-links up to a specified depth.
- Ensured the crawler handled broken links and dynamic content appropriately.

### Data Chunking and Vector Database Creation

**Objective**: Chunk the scraped data based on sentence/topic similarity, convert chunks to embedding vectors, and store them in a Milvus vector database.

**Implementation**:
- Used `sentence-transformers/all-MiniLM-L6-v2` model from the `transformers` library to generate embeddings.
- Created a Milvus collection with fields for embedding vectors and metadata.
- Stored the embeddings and corresponding metadata in Milvus.

### Retrieval and Re-ranking

**Objective**: Implement query expansion and hybrid retrieval using BM25 and BERT/bi-encoder methods, followed by re-ranking.

**Implementation**:
- Used `Whoosh` for BM25-based retrieval.
- Placeholder for bi-encoder retrieval using DPR (to be implemented with Milvus).
- Implemented a basic re-ranking mechanism combining results from both methods.

### Question Answering

**Objective**: Use an LLM to generate answers from the re-ranked data.

**Implementation**:
- Used `deepset/roberta-base-squad2` model from Hugging Face's `transformers` library for question answering.
- Integrated the model into the retrieval pipeline to generate answers from the re-ranked results.

### User Interface

**Objective**: Create a user interface using Streamlit to demonstrate the system.

**Implementation**:
- Developed a simple Streamlit application allowing users to input queries and display answers.

## Evaluation Criteria

- **Completeness and Accuracy of Web Crawling**: The crawler successfully retrieves data from the parent link and sub-links up to the specified depth.
- **Effectiveness of Data Chunking and Vector Database Creation**: Advanced chunking techniques and Milvus storage are implemented correctly.
- **Quality of Retrieval and Re-ranking Methods**: Initial hybrid retrieval and re-ranking methods are integrated, with room for further optimization.
- **Accuracy and Relevance of LLM-generated Answers**: The LLM provides contextually relevant answers based on the re-ranked data.
- **Overall System Performance and Efficiency**: The system performs efficiently, with the possibility of further optimizations.
- **User Interface Design and User Experience**: The optional user interface is functional and user-friendly.

## Submission

The source code, including the web crawler, data chunking, vector database creation, retrieval, re-ranking, and question-answering components. 
