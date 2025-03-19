import time
import sys
import warnings
import json
import pandas as pd
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain.vectorstores import Chroma
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain.schema import Document
from huggingface_hub import notebook_login

notebook_login("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")

# LOAD VECTOR STORE
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
directory = "./vector__store"
vector_store = Chroma(persist_directory=directory, embedding_function=embeddings)
vector_store.get()

# LOAD MODEL AND PROMPT
prompt = hub.pull("rlm/rag-prompt")

Llama_model = "meta-llama/Llama-3.2-1B-Instruct"

llm = HuggingFaceEndpoint(repo_id=Llama_model,
                            task="text-generation",
                            max_new_tokens=1024,
                            do_sample=False,
                            repetition_penalty=1.03)

# GET LIST OF COURSES 
with open("courses.txt", "r") as f:
  courses = f.read().splitlines()

def classify_question(question: str):
  if "between" in question:
    filters = []
    for c in courses:
      if c in question:
        filters.append({"class_name":{"$eq": c}})
    filter = {"$or": filters}
    return filter if len(filters) != 0 else None
  elif "require" in question or "have" in question:
    if "corequisite" in question and "prerequisite" in question:
      coreq_pos = question.find("corequisite")
      prereq_pos = question.find("prerequisite")
      filters = []
      if coreq_pos < prereq_pos:
        sec_1 = question[:coreq_pos]
        sec_2 = question[coreq_pos:]
        i = 1
        j = 1
        for c in courses:
          if c in sec_1:
            filters.append({f"coreq_{i}": {"$eq": c}})
            i += 1
          if c in sec_2:
            filters.append({f"prereq_{j}": {"$eq": c}})
            j += 1
        filter = {"$and": filters} if "and" in question else {"$or": filters}
        if len(filters) < 2:
          filter = filters
        return filter[0] if len(filters) != 0 else None
      else:
        sec_1 = question[:prereq_pos]
        sec_2 = question[prereq_pos:]
        for c in courses:
          i = 1
          j = 1
          if c in sec_1:
            filters.append({f"prereq_{i}": {"$eq": c}})
            i += 1
          if c in sec_2:
            filters.append({f"coreq_{j}": {"$eq": c}})
            j += 1
        filter = {"$and": filters} if "and" in question else {"$or": filters}
        if len(filters) < 2:
          filter = filters
        return filter[0] if len(filters) != 0 else None

    # PREREQ
    elif "prerequisite" in question:
      # Require multiple prerequisites
        filters = []
        i = 1
        for c in courses:
          if c in question:
            filters.append({f"prereq_{i}": {"$eq": c}})
            i += 1
        filter = {"$and": filters} if "and" in question else {"$or": filters}
        if len(filters) < 2:
          filter = filters
        return filter[0] if len(filters) != 0 else None
    else: 
      filters = []
      i = 1
      for c in courses:
        if c in question:
          filters.append({f"coreq_{i}": {"$eq": c}})
          i += 1
      filter = {"$and": filters} if "and" in question else {"$or": filters}
      if len(filters) < 2:
        filter = filters
      return filter[0] if len(filters) !=0 else None
  elif "need" in question:
    last_course = ""
    for c in courses:
      if c in question:
        if question.find(last_course) < question.find(c):
          last_course = c
    return {"class_name": last_course}
  else:
    #FIND THE CLASS
    for c in courses:
      if c in question:
        return {"class_name": c}
  return None


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    filter = classify_question(state["question"])
    retrieved_docs = vector_store.similarity_search(state["question"],
        k=5,
        filter=filter)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# QA HERE
response = graph.invoke({"question": q})
print(response["answer"])