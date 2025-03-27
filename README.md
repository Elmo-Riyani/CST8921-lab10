# CST8921 – Cloud Industry Trends  
## Lab 10 – Setting Up Azure Cognitive Search Index and Deploying Embedding & LLM Models

---

### **Student Name:** [Your Name]  
### **Date:** [Insert Date]

---

## 🧠 Introduction

In this lab, we explored the use of AI and ML services in cloud environments. We set up an Azure Cognitive Search index and deployed embedding and LLM models using Azure OpenAI to build a Retrieval Augmented Generation (RAG) system. The goal was to enable document-based question answering using LangChain, combining indexing, embeddings, and language models.

---

## 🎯 Objective

Set up Azure Cognitive Search Index and deploy Embedding & Large Language Models (LLM) on Azure OpenAI for document-based Q&A using Retrieval Augmented Generation (RAG).

---

## 🔧 Prerequisites

- Cloud portal access with Azure  
- Basic understanding of cloud AI and ML  
- Web browser, VS Code, Python 3.10+, and internet connection

---

## 🧪 Lab Activity Overview

### Part 1: Setting Up Azure Cognitive Search Index

#### ✅ Step 1: Create an Azure AI Search Service

1. Logged in to Azure Portal
2. Created a new resource → Azure AI Search
3. Filled in:
   - Subscription
   - Resource Group
   - Search Service Name
   - Region
   - Pricing Tier (Free)
4. Clicked **Review + Create**, then **Create**
5. After deployment, accessed the resource

![Step 1](Screenshots/1.PNG)
![Step 2](Screenshots/2.PNG)
![Step 3](Screenshots/3.PNG)

#### ✅ Step 2: Create an Index in Azure AI Search

1. Navigated to **Indexes** → **Add Index**
2. Created:
   - Field 1: `data` (retrievable, searchable)
   - Field 2: `source` (retrievable, searchable)
3. Clicked **Create**

![Step 1](Screenshots/4.PNG)
![Step 2](Screenshots/5.PNG)
![Step 3](Screenshots/6.PNG)
![Step 1](Screenshots/7.PNG)
![Step 2](Screenshots/8.PNG)

#### ✅ Step 3: Retrieve Endpoint & Key

- Collected Endpoint URL from **Overview**
- Copied Primary Admin Key from **Keys**
- Saved for scripting

![Step 2](Screenshots/8.PNG)
![Step 2](Screenshots/9.PNG)

---

### Part 2: Deploying Embedding and LLM Models on Azure OpenAI

#### ✅ Step 1: Create an Azure OpenAI Service

1. Created a new resource: Azure OpenAI
2. Selected networking options, skipped tags
3. Clicked **Review + Create** → Deployed and accessed the resource

![Step 1](Screenshots/10.PNG)
![Step 2](Screenshots/11.PNG)
![Step 3](Screenshots/12.PNG)
![Step 1](Screenshots/13.PNG)

#### ✅ Step 2: Deploy Embedding and LLM Models

1. Deployed:
   - **text-embedding-ada-002** → named `embedding-model`
   - **gpt-3.5-turbo-instruct** → named `llm-model`
2. Saved deployment names

![Step 1](Screenshots/14.PNG)
![Step 2](Screenshots/15.PNG)
![Step 3](Screenshots/17.PNG)
![Step 1](Screenshots/18.PNG)
![Step 2](Screenshots/19.PNG)
![Step 1](Screenshots/20.PNG)
![Step 2](Screenshots/21.PNG)
![Step 3](Screenshots/22.PNG)

#### ✅ Step 3: Retrieve Endpoint & Key

- Retrieved API key and endpoint from **Keys and Endpoints**
- Stored for scripting

---

### Part 3: Scripting with LangChain for RAG (Using VS Code)

#### ✅ Step 1: Install Required Libraries

```bash
pip install azure-search-documents langchain openai pypdf tiktoken unstructured langchain-openai langchain-community
```

![Step 1](Screenshots/23.PNG)

#### ✅ Step 2: Import Required Libraries

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import os
```

![Step 1](Screenshots/24.PNG)

#### ✅ Step 3: Configure Azure Cognitive Search Client

```python
index_name = "azure-rag-demo-index"
endpoint = "YOUR_AZURE_SEARCH_ENDPOINT"
key = "YOUR_AZURE_SEARCH_KEY"

credential = AzureKeyCredential(key)
client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
```

![Step 1](Screenshots/25.PNG)

#### ✅ Step 4: Configure LLM Model

```python
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY"
os.environ["AZURE_OPENAI_ENDPOINT"] = "YOUR_OPENAI_ENDPOINT"

llm = AzureOpenAI(deployment_name="YOUR_LLM_DEPLOYMENT_NAME", model="gpt-3.5-turbo-instruct", temperature=1)
```
![Step 1](Screenshots/26.PNG)

#### ✅ Step 5: Load and Process PDF Data

Downloaded:
[Deep Learning Applications and Challenges in Big Data Analytics](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-014-0007-7)

```python
pdf_link = "demo_paper.pdf"
loader = PyPDFLoader(pdf_link, extract_images=False)
data = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 5000,
    chunk_overlap = 20,
    length_function = len
)
chunks = text_splitter.split_documents(data)
```

![Step 1](Screenshots/27.PNG)

#### ✅ Step 6: Store Data in Azure Search Index

```python
for index, chunk in enumerate(chunks):
    data = {
        "id" : str(index + 1),
        "data" : chunk.page_content,
        "source": chunk.metadata["source"]
    }
    result = client.upload_documents(documents=[data])
```

![Step 1](Screenshots/28.PNG)

#### ✅ Step 7: Create Function for Retrieval-Augmented Generation

```python
def generate_response(user_question):
    context = 
    results = client.search(search_text=user_question, top=2)
    for doc in results:
        context += "
" + doc['data']

    qna_prompt_template = f"""You will be provided with the question and a related context, you need to answer the question using the context.

Context:
{context}

Question:
{user_question}

Make sure to answer the question only using the context provided, if the context doesn't contain the answer then return \"I don't have enough information to answer the question\".

Answer:"""

    response = llm.invoke(qna_prompt_template)
    return response
```

![Step 1](Screenshots/29.PNG)

#### ✅ Step 8: Test the System

```python
user_question = "What is deep learning?"
response = generate_response(user_question)
print("Answer:", response)
```
![Step 1](Screenshots/30.PNG)


✅ **Sample Output:**
```
Answer: Deep learning is a type of machine learning that uses algorithms to train high-level data representations and patterns from large amounts of data.
```
![Step 1](Screenshots/32.PNG)


⚠️ **Note:** 
    
- Install the langchain-community module:
- 🛠 Run this in your terminal:
```
  pip install -U langchain-community
```
![Step 1](Screenshots/31.PNG)

🔁 Then Update Your Code

After installing, you should update the import for PyPDFLoader to use the new module path:
✏ Replace this line:
```
from langchain.document_loaders import PyPDFLoader
```
✅ With this:
```
from langchain_community.document_loaders import PyPDFLoader
```
- A deprecation warning was resolved by using `.invoke()` instead of calling the LLM directly.

---

## ✅ Important Notes

- All components were successfully integrated and tested in VS Code
- Project structured in folder `CST8921-Lab10`


---

## 📎 Attachments

- Screenshot of Azure portal (Cognitive Search, OpenAI service)
- Screenshot of VS Code terminal with output
- `main.py` file (optional for submission)

---

## 📚 References

- [Azure Cognitive Search Documentation](https://learn.microsoft.com/en-us/azure/search/)
- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/)
- [LangChain Documentation](https://docs.langchain.com/)
- [Source PDF](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-014-0007-7)
