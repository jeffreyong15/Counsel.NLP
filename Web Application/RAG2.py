import time
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.vectorstores import Chroma
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain.schema import Document
from huggingface_hub import login
import nltk
import requests

nltk.download('all', quiet=True)

login("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")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
directory = "./vector__store"
vector_store = Chroma(persist_directory=directory, embedding_function=embeddings)
vector_store.get()

prompt = hub.pull("rlm/rag-prompt")

model_name = "meta-llama/Llama-3.2-1B-Instruct"

# Download model and tokenizer
cache_dir = "./models/"  # Saves model locally
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="cpu")#, low_cpu_mem_usage=True)
llama_pipeline = pipeline("text-generation",
                          model=model,
                          tokenizer=tokenizer,
                          device_map="cpu",
                          return_full_text=False,
                          max_new_tokens=512,
                          do_sample=True,
                          top_p=0.95,
                          temperature=0.7)

llm = HuggingFacePipeline(pipeline=llama_pipeline)

file_path = "courses.txt"
with open(file_path, "r") as f:
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
    source_documents: List[str]

def retrieve(state: State) -> State:
    filtered = classify_question(state["question"])
    retrieved_docs = vector_store.similarity_search(
        state["question"],
        k=3,
        filter=filtered
    )
    # Extract source document content
    source_documents = [doc.page_content for doc in retrieved_docs]
    return {"context": retrieved_docs, "source_documents": source_documents}

def generate(state: State) -> State:
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = f"""### System:
    You are an academic advising assistant. Answer the student's question directly and completely, but do not include extra information beyond what was asked.

    ### User:
    Question: {state['question']}

    ### Context:
    {docs_content}

    ### Answer:
    """
    response = llm.invoke(messages)
    return {"answer": response, "source_documents": state["source_documents"]}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
print("Graph Built")
def get_chatbot_response(user_question):
  retries = 3  # Number of retry attempts
  for attempt in range(retries):
    try:
      response = graph.invoke({"question": user_question})
      return response["answer"].strip()
    except requests.exceptions.HTTPError as e:
      if e.response.status_code == 503:  # Handle server unavailability
        print(f"Server unavailable. Retrying ({attempt + 1}/{retries})...")
        time.sleep(1)  # Wait before retrying
      else:
        raise e  # Re-raise the error if it's not a 503
  return "Error: The chatbot service is temporarily unavailable. Please try again later."