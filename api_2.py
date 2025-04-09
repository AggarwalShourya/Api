from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
import faiss

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Load and prepare dataset
df = pd.read_csv("Data_test.csv")
df["combined"] = df.apply(lambda row: f"""Title: {row['title']}
Description: {row['description']}
Assessment Length: {row['Assesment Length']}
Job Level: {row['job level']}
Languages: {row['languages']}""", axis=1)

# Set up embeddings and FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

documents = [
    Document(page_content=row["combined"], metadata={
        "title": row["title"],
        "url": row["Url"],
        "description": row["description"],
        "duration": row["Assesment Length"],
        "job level": row["job level"],
        "languages": row["languages"],
        "adaptive": row.get("adaptive", "No"),
        "remote_testing": row.get("remote_testing", "No"),
    }) for _, row in df.iterrows()
]

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(dict(enumerate(documents))),
    index_to_docstore_id={i: i for i in range(len(documents))},
)
vector_store.add_documents(documents)
retriever = vector_store.as_retriever()

# LLM configuration
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.6,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

system_prompt = """You are a helpful assistant who recommends SHL assessments to users.
Only suggest from the retrieved context. Return at most 10 assessments.
Context:
{context}
Question: {question}"""

prompt = ChatPromptTemplate.from_template(system_prompt)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# FastAPI App
app = FastAPI()

class Query(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/recommend")
def recommend(query: Query):
    relevant_docs = retriever.get_relevant_documents(query.query)[:10]
    
    results = []
    for doc in relevant_docs:
        meta = doc.metadata
        results.append({
            "title":meta.get("title"),
            "job level": meta.get("job level"),
            "url": meta.get("url", ""),
            "adaptive_support": meta.get("adaptive", "No"),
            "remote_support": meta.get("remote_testing", "No"),
            "description": meta.get("description", ""),
            "duration": int(''.join(filter(str.isdigit, str(meta.get("duration", "0")))))
        })
    
    return {"recommended_assessments": results}
