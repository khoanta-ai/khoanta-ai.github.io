---
layout: post
title: "[llm-zoomcamp]-intro-01: Introduction to LLMs and RAG"
description: In this module, we will learn what LLM and RAG are and implement a simple RAG pipeline to answer questions about the FAQ Documents from our Zoomcamp courses
tags: llms rag datatalks.club llm-zoomcamp
date: 2024-06-22
featured: true
toc:
  sidebar: left
disqus_comments: true

---
<span style='color:red'>This is my first experience with the RAG system. If there are any errors, whether in terminology, structure, or understanding, ..., please inform me.</span>

I will share knowledge that I have learn about the first module in [llm-zoomcamp course](https://github.com/DataTalksClub/llm-zoomcamp/tree/main) from [DataTalks.Club](https://datatalks.club/)

In this article I will discus about implementing simple Retrieval Augmented Generation (RAG) pipeline to make a Q&A system. This Q&A system can answer questions about the FAQ Documents from the Zoomcamp courses of [DataTalks.Club](https://datatalks.club/). This is just basic knowledge to make a Q&A system base on libraries in Python, and we do not discus about the theory.

<!-- ## Overview
- Overview Simple RAG pipline
- Preparing the Environment
- Build Essential Functions
  - Indexing documents
  - Retrieval
  - Generation with LLMs (OpenAI API and Ollama)
- Homework -->

## Simple RAG pipline
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/posts/llm_zoomcamp/assets/Simple_RAG_Pipeline_Module1.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A simple RAG Pipline.
</div>

A simple RAG pipeline might have the following components and stages below
- Components:
  - **User:** Who has a question and want to get the answer for this question. (e.g students).
  - **Database-DB (Knowledge Base - KB):** It serves as a respository of information (documents) that the system can query to retrieve relevant document or data. (e.g indexed docuemts contained the questions and corresponding answers of the courses)
  - **Large Language Model (LLM):** A language model trained on vast amounts of text data, enabling it to perform a wide range of language-related tasks. Receive the a **prompt** and generate more informed and accurate response (answer).
    - **Prompt:**  In the scope, it has two elements: user's question and the context (which is retrieved from the DB). It serves as an input of LLM.
- Generally, there are three stages in a simple RAG pipeline:
  - Stage 0: We need to build **DB or KB** from the documents and **search engine**. Moreover, we also prepare a **prompt template** to send it to LLM .
  - <span style="color:#9673A6">Stage1 (violet):</span> Initially, A user poses a **question**, then sends the question to a built-in **search engine** within the **database**. The **search engine** responds with the **top corresponding contexts** from the **database** to the user.
  - <span style="color:red">Stage2 (red):</span> Subsequently, the system constructs a **prompt** using the **template**, which includes **instructions for the LLM** along with the **question** and the **context** retrieved from Stage 1.
  - <span style="color:#01A88D">Stage3 (green):</span> Finally, this **prompt** is forwarded to the LLM, which then generates an **answer** that is delivered back to the user.

## Preparing the Environment
I use codespaces of Github (you can use anaconda or miniconda). Click on the `<> Code` button $$\rightarrow$$ `Codespace` tag $$\rightarrow$$ new tab is opened and we choose `Open in VS Code Desktop`

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/posts/llm_zoomcamp/assets/env00.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/posts/llm_zoomcamp/assets/env01.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Use Codespace of Github with Visual Studio Code
</div>

Then we run the command below in the terminal:
```
pip install tqdm notebook==7.1.2 openai elasticsearch pandas scikit-learn
```

For the remainder of this article, we will utilize Jupyter Notebook for coding.

## Build Preparation Functions
### Preparing Documents Function
Firstly, we download the documents and we will format this to generate a DB or KB.

```bash
## Download documents.json
!wget https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-intro/documents.json
```
The format of `documents.json` is as follows:

```
Example JSON structure:
    [
        {
            "course": "Course Name",
            "documents": [
                {
                    "text": "Document text",
                    "question": "Question related to the document",
                    "section": "Section of the course the document belongs to"
                },
                ...
            ]
        },
        ...
    ]
```
`build_documents_from_json(json_path)` function converts documents from a JSON format into a list. Each element in the list is a dictionary containing four keys: 'course', 'text', 'question', and 'section'. To achieve this, the function reads the JSON file specified by json_path. It iterates through each course in the JSON, and for each document within a course, it adds the 'course' information from the course dictionary to the document. Finally, it appends each document to a list, which is then returned.

```python
def build_documents_from_json(json_path):
    '''
    Convert document json format to a list of
    elements which contain 4 objects: 
    course, text, question, and section
    '''
    with open(json_path, 'rt') as f_in:
        docs_raw = json.load(f_in) 
    documents = []
    
    for course_dict in docs_raw:
        for doc in course_dict['documents']:
            doc['course'] = course_dict['course']
            documents.append(doc)
    return documents
```
This function will return a list of dictionary as below:
```
[
  {
    "course": "Course Name",
    "text": "Document text",
    "question": "Associated question",
    "section": "Document section"
  },
  [
  {
    "course": "Course Name",
    "text": "Document text",
    "question": "Associated question",
    "section": "Document section"
  },
  ...
]
]
```
### Configuring Search Engine and Indexing Documents Function
In this section, we aim to develop a search engine designed to retrieve contexts relevant to the user's queries.

#### Minsearch
Minseatch is a simple search index using TF-IDF and cosine similarity for text fields and exact matching for keyword fields.

The provided Python function build_minsearch initializes a search index with specified text and keyword fields, then inxex it with documents to make them searchable.

```python
def build_minsearch(documents, text_fields, keyword_fields):
    # SELECT * WHERE course = 'data-engineering-zoomcamp';
    index = minsearch.Index(
        text_fields=text_fields,
        keyword_fields=keyword_fields
    )
    index.fit(documents)
    return index
```

#### Elasticsearch
The provided code snippet defines two functions for working with Elasticsearch in Python:

`es_create_index_if_not_exists`: This function attempts to create an Elasticsearch index with a given name and configuration. If the index already exists (indicated by a resource_already_exists_exception), the error is ignored, and the function proceeds to index the provided documents. Any other exceptions are raised.

`build_elastic_search`: This function initializes an Elasticsearch client, sets up an index with specified settings, and populates it with documents. It uses the `es_create_index_if_not_exists function` to ensure the index is created only if it doesn't already exist, thus avoiding duplication errors.

```python
def es_create_index_if_not_exists(es, index_name, body, documents):
    """Create the given ElasticSearch index and ignore error if it already exists"""
    try:
        es.indices.create(index=index_name, body=body)
        for doc in tqdm(documents):
            es.index(index=index_name, document=doc)
    except elasticsearch.exceptions.RequestError as ex:
        if ex.error == 'resource_already_exists_exception':
            pass # Index already exists. Ignore.
        else: # Other exception - raise it
            raise ex


def build_elastic_search(elasticsearch_url, documents, index_name="course-questions"):
    index_settings = config_elastic_search()
    es_client = Elasticsearch(elasticsearch_url) 
    # es_client.indices.create(index=index_name, body=index_settings)
    es_create_index_if_not_exists(es=es_client, index_name=index_name, 
                                  body=index_settings, documents=documents)
    return es_client
```
### Prompt Template Function
Create a prompt template for use with a LLM. Each time a query is submitted to the LLM, incorporate the user's question and the context retrieved from the search engine into this template.

```python
def build_prompt(query, search_results):
    prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.
QUESTION: {question}
CONTEXT: {context}
""".strip()
    
    context = ""
    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt
```

### Invoking OpenAI API function with Ollama
Use Ollama with OpenAI API

```python
def build_llm(base_url, api_key):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    return client
```

## Build Essential Functions
We need a function to index the processed documents above (we can use minsearch or elastic search). Subsequently we build a search function for the search engine and

### Retrieval
#### Minsearch
The `minsearch_search` function conducts a search using a specified search engine, applying filters and boosts to the query, and limits the results to a specified number. It returns the search results.

```bash
## Download minsearch.py
!wget https://raw.githubusercontent.com/alexeygrigorev/minsearch/main/minsearch.py
```
```python
def minsearch_search(query, search_engine, filter_dict, boost_dict, num_results):
    results = search_engine.search(
        query=query,
        filter_dict=filter_dict,
        boost_dict=boost_dict,
        num_results=num_results
    )
    return results
```
#### Elasticseatch

The `elastic_search` function receives an index_name (corresponding to a pre-configured index), a configured elastic_query, and an instance of Elasticsearch, then returns the search results.

```python
def elastic_search(index_name, elastic_query, es_client):
    response = es_client.search(index=index_name, body=elastic_query)
    result_docs = []
    for hit in tqdm(response['hits']['hits']):
        result_docs.append(hit['_source'])
    return result_docs
```
### Generation with LLMs (OpenAI API with Ollama)
The `query_llm` function is a Python function designed to query a LLM using a specified client interface. It accepts three parameters: prompt, which is the text input or question to be submitted to the model; client, an instance of the API client that facilitates communication with the LLM service; and model_name, the identifier of the specific language model to be queried. The function works by sending a request to the LLM, structured as a message with the user's role and the provided prompt. Upon receiving the response, it extracts and returns the content of the first message from the model's choices. This function encapsulates the interaction with an LLM, making it straightforward to integrate language model capabilities into various applications or services, allowing for dynamic and intelligent text generation based on user inputs.

```python
def query_llm(prompt, client, model_name):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{'role':'user', 'content':prompt}]
    )
    return response.choices[0].message.content
```
### Run RAG
Finally, we encapsulate three functions corresponding to the three stages depicted in Figure 1 to process a user query within the RAG system. An example how to use the whole code also provided in the subsections below.
#### RAG with Minsearch
**Minsearch Rag Function**
```python
def minsearch_rag(query):
    minsearch_results = minsearch_search(query=query, search_engine=minsearch_engine, 
                       filter_dict=filter_dict, boost_dict=boost_dict, 
                       num_results=num_results)
    prompt = build_prompt(query=query, search_results=minsearch_results)
    response_res = query_llm(prompt=prompt, client=phi3_client, model_name=model_name)
    return response_res
```
Run RAG with Minsearch
```python
json_doc_path = 'documents.json'
cvt_documents = build_documents_from_json(json_doc_path)
# print(cvt_documents)

text_fields = ["question", "text", "section"]
keyword_fields = ["course"]
minsearch_engine = build_minsearch(cvt_documents, text_fields, keyword_fields)

filter_dict = {'course': 'data-engineering-zoomcamp'}
boost_dict = {'question': 3.0, 'section': 0.5}
num_results = 5

base_url = 'http://localhost:11434/v1/'
api_key = 'ollama'
model_name = 'phi3'
phi3_client = build_llm(base_url, api_key)

minsearch_rag(query= 'the course has already started, can I still enroll?')
```
#### RAG with Elasticsearch
**Elasticsearch RAG Function**
```python
def elastic_rag(query):
    elastic_query = build_search_query(num_results=num_results, query=query, 
                                       text_boost_fields=text_boost_fields,
                                       query_type=query_type,filter_dict=filter_dict)
    elastic_results = elastic_search(index_name, elastic_query, es_client)
    prompt = build_prompt(query=query, search_results=elastic_results)
    response_res = query_llm(prompt=prompt, client=phi3_client, model_name=model_name)
    return response_res
```
Run Elasticsearch with RAG
```python
json_doc_path = 'documents.json'
cvt_documents = build_documents_from_json(json_doc_path)
# print(cvt_documents)

elasticsearch_url = 'http://localhost:9200'
index_name = "course-questions2"
es_client = build_elastic_search(elasticsearch_url, cvt_documents, index_name)

num_results = 10
text_boost_fields = ["question^3", "text", "section"]
query_type = "best_fields"
# keyword_fields = ["course"]
filter_dict = {'course': 'data-engineering-zoomcamp'}
elastic_query = build_search_query(num_results=num_results, query=query, 
                                       text_boost_fields=text_boost_fields,
                                       query_type=query_type,filter_dict=filter_dict)

prompt = build_prompt(query=query, search_results=elastic_results)
base_url = 'http://localhost:11434/v1/'
api_key = 'ollama'
model_name = 'phi3'
phi3_client = build_llm(base_url, api_key)

print(elastic_rag(query= 'the course has already started, can I still enroll?'))
```

### Whole pipeline
The complete code is available at the following links:

#### Simple RAG with Minsearch
[https://github.com/khoanta-ai/llm_zoomcamp/blob/main/01-intro/Simple_RAG_minsearch_clean.ipynb](https://github.com/khoanta-ai/llm_zoomcamp/blob/main/01-intro/Simple_RAG_minsearch_clean.ipynb)

#### Simple RAG with Elasticsearch
[https://github.com/khoanta-ai/llm_zoomcamp/blob/main/01-intro/Simple_RAG_elasticsearch_clean.ipynb](https://github.com/khoanta-ai/llm_zoomcamp/blob/main/01-intro/Simple_RAG_elasticsearch_clean.ipynb)

## Other Information
### Ollama - Running LLMs on a CPU
#### Docker
```
docker run -it \
    -v ollama:/root/.ollama \
    -p 11434:11434 \
    --name ollama \
    ollama/ollama
```

#### Forward a port
```
- Check the port in '-p 11434:11434', we get port after ':'.
- In Visual Studio Code, in the terminal, choose 'PORTS' tag, click 'Forward a Port' then add the '11434' port.
- Use command "docker ps" to find 'NAMES' of the ollama container.
```

#### Pulling the model
```
docker exec -it ollama bash
ollama pull phi3
```

#### Testing
```bash
curl http://localhost:11434/api/chat -d '{
  "model": "phi3",
  "messages": [
    { "role": "user", "content": "why is the sky blue?" }
  ]
}'
```

### ElasticSearch
#### Run ElasticSearch with Docker
```bash
docker run -it \
    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3
```
#### Index settings:
```
{
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"} 
        }
    }
}
```
#### Query:
```
{
    "size": 5,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["question^3", "text", "section"],
                    "type": "best_fields"
                }
            },
            "filter": {
                "term": {
                    "course": "data-engineering-zoomcamp"
                }
            }
        }
    }
}
```
#### Configuration ElasticSearch Function
```python
def config_elastic_search():
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"} 
            }
        }
    }
    return index_settings
```
#### Building Search Query Elasticsearch Function
```python
def build_search_query(num_results, query, text_boost_fields, query_type, filter_dict):
    search_query = {
        "size": num_results,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": text_boost_fields,
                        "type": query_type
                    }
                },
                "filter": {
                    "term": filter_dict
                }
            }
        }
    }

    return search_query
```



<!-- In my latest blog post, I delve into my initial encounter with the Retrieval Augmented Generation (RAG) system, a journey marked by curiosity and a quest for understanding. This exploration is guided by the first module of the llm-zoomcamp course offered by DataTalks.Club, a resource I found invaluable in navigating the complexities of RAG. My aim is to share the insights and knowledge I've gained, focusing on the practical application of RAG to develop a foundational Q&A system. This system is designed to address the FAQs of the Zoomcamp courses, leveraging Python libraries to provide answers. It's a foray into the practical without delving deep into the theoretical underpinnings of RAG systems.

In my latest blog post, I discuss my first experience with the Retrieval Augmented Generation (RAG) system. It all began with the first module of the llm-zoomcamp course by DataTalks.Club, which proved invaluable in helping me understand RAG's complexities. My goal is to share practical insights into how RAG can be used to build a Q&A system that addresses common questions from Zoomcamp courses. I've been leveraging Python libraries to develop this system, focusing on real-world applications rather than diving into theoretical details. -->

