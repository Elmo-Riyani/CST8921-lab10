from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import os

# Azure Cognitive Search config
index_name = "azure-rag-demo-index"
endpoint = "https://search-lab10.search.windows.net"
key = "zVm0vxJ8dl0QkU47wI0s2PMFjjpv8jXKNBosSBdbabAzSeDOCtCM"

credential = AzureKeyCredential(key)
client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

# Azure OpenAI config
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_KEY"] = "4QlYTTvZHSdQfAkWny18wFg3ydZAFzfQ9F7yEyOOASj9bkmitfmTJQQJ99BCACfhMk5XJ3w3AAAAACOGIzTw"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://riya0-m8qxb6a6-swedencentral.openai.azure.com/"

llm = AzureOpenAI(
    deployment_name="llm-lab10",
    model="gpt-3.5-turbo-instruct",
    temperature=1
)

# Load PDF and split text
pdf_link = "demo_paper.pdf"
loader = PyPDFLoader(pdf_link, extract_images=False)
data = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 5000,
    chunk_overlap = 20,
    length_function = len
)
chunks = text_splitter.split_documents(data)

# Upload chunks to Azure Search
for index, chunk in enumerate(chunks):
    data = {
        "id" : str(index + 1),
        "data" : chunk.page_content,
        "source": chunk.metadata["source"]
    }
    result = client.upload_documents(documents=[data])

# Define Retrieval Q&A function
def generate_response(user_question):
    context = """"""
    results = client.search(search_text=user_question, top=2)
    for doc in results:
        context += "\n" + doc['data']

    qna_prompt_template = f"""You will be provided with the question and a related context, you need to answer the question using the context.

Context:
{context}

Question:
{user_question}

Make sure to answer the question only using the context provided, if the context doesn't contain the answer then return "I don't have enough information to answer the question".

Answer:"""

    response = llm.invoke(qna_prompt_template)
    return response

# Ask a test question
user_question = "What are some challenges in big data analytics??"
response = generate_response(user_question)
print("Answer:", response)
