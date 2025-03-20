import time
import json
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
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

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    if not json_data:
        raise ValueError("JSON data is empty")

    print(f"Successfully loaded {len(json_data)} courses")
    return json_data

dataset_path = "SJSU_courses_with_metadata_updated.json"
gen_path = "complete_Gen_Advising.json"
SJSU_dataset = load_json_data(dataset_path)
Gen_Advising_Dataset = load_json_data(gen_path)

# Process dataset (Courses & Majors)
class_mapping = {}
code = ["No prerequisites listed", "No corequisites listed"]
majors = []
category = []
def process_data(json_data, gen_data):
    documents = []
    for item in gen_data:
        content = [
            f"Title: {item.get('title', 'N/A')}",
            f"Description: {item.get('description', 'N/A')}"
        ]
        doc = Document(
            page_content="\n".join(content),
            metadata={"title": item.get('title', 'N/A')}
        )
        documents.append(doc)


    for item in json_data:
        majors.append(item['metadata']['major']) if item['metadata']['major'] not in majors else None
        if item['id'].isdigit():
            category.append(item['metadata']['category']) if item['metadata']['category'] not in category else None
        title = item.get('title', 'N/A')
        if title != "N/A":
            class_name = title.split("-")[0].strip()
            code.append(class_name)
            class_mapping[class_name] = title
        content = [
            f"Title: {item.get('title', 'N/A')}",
            f"Type: {'Major' if 'core_courses' in item else 'Course'}",
            f"Units: {item.get('units', 'N/A')}",
            f"Description: {item.get('description', 'N/A')}",
            f"Grading: {item.get('grading', 'N/A')}",
            f"Class Structure: {item.get('class_structure', 'Class structure not found')}"
        ]

        # Handle prerequisites & corequisites
        if item.get('prerequisite(s)'):
            content.append("Prerequisite(s): " + ", ".join(item['prerequisite(s)']))

        if item.get('corequisite(s)'):
            content.append("Corequisite(s): " + ", ".join(item['corequisite(s)']))

        if item.get('pre/corequisite(s)'):
            content.append("Pre/Corequisite(s): " + ", ".join(item['pre/corequisite(s)']))

        if item.get('notes'):
            content.append("Note(s): " + ", ".join(item['notes']))

        # Handle core courses
        if 'core_courses' in item:
            content.append("\nCore Courses:")
            for course in item.get('core_courses', []):
                content.append(f"- {course['course']}: {course['title']} ({course['units']} units)")

        # Handle specialization tracks
        if 'specialization_tracks' in item:
            content.append("\nSpecialization Tracks:")

            for specialization, details in item['specialization_tracks'].items():
                content.append(f"\n- {specialization}:")

                if isinstance(details, list):  # MSAI-style specialization (direct list of courses)
                    for course in details:
                        content.append(f"  - {course['course']}: {course['title']} ({course['units']} units)")

                elif isinstance(details, dict):  # MSSE-style specialization (nested dictionary)
                    if 'overview' in details:
                        content.append(f"  Overview: {details['overview']}")

                    if 'required_core_courses' in details:
                        content.append("\n  Required Core Courses:")
                        for course in details['required_core_courses']:
                            content.append(f"    - {course['course']}: {course['title']} ({course['units']} units)")

                    if 'specialization_choice_courses' in details:
                        content.append("\n  Specialization Choice Courses:")
                        for course in details['specialization_choice_courses']:
                            content.append(f"    - {course['course']}: {course['title']} ({course['units']} units)")

        # Handle elective courses
        if 'elective_courses' in item:
            content.append("\nElective Courses:")
            if 'overview' in item['elective_courses']:
                content.append(f"  Overview: {item['elective_courses']['overview']}")
                if 'restricted_courses' in item['elective_courses']:
                    content.append("\n  Restricted Courses (cannot be taken as electives):")
                    for course in item['elective_courses']['restricted_courses']:
                        if isinstance(course, dict):
                            content.append(f"    - {course['course']}: {course['title']} ({course['units']} units)")
                        elif isinstance(course, str):
                            content.append(f"    - {course}")
            else:
                for area, courses in item['elective_courses'].items():
                    content.append(f"\n- {area}:")
                    for course in courses:
                        if isinstance(course, dict):
                            content.append(f"  - {course['course']}: {course['title']} ({course['units']} units)")
                        elif isinstance(course, str):
                            content.append(f"  - {course}")

        # Handle graduate writing requirement
        if 'graduate_writing_requirement' in item:
            content.append("\nGraduate Writing Requirement:")
            gww = item['graduate_writing_requirement']
            if 'courses' in gww:  # Multi-course format
                for course in gww['courses']:
                    content.append(f"  - {course['course']}: {course['title']} ({course['units']} units)")
                    if 'description' in course:
                        content.append(f"    Description: {course['description']}")
            elif 'course' in gww:  # Single-course format
                content.append(f"  - {gww['course']}: {gww['title']} ({gww['units']} units)")

        # Handle culminating experience
        if 'culminating_experience' in item:
            content.append("\nCulminating Experience Options:")
            for option, courses in item['culminating_experience'].items():
                content.append(f"\n- {option}:")
                for course in courses:
                    if isinstance(course, dict):
                        content.append(f"  - {course['course']}: {course['title']} ({course['units']} units)")
                    elif isinstance(course, str):
                        content.append(f"  - {course}")


        doc = Document(
        page_content="\n".join(content),
        metadata={"title": item.get('title', 'N/A'),
                  "class_name": class_name if item['id'].isdigit() else 'N/A',
                  "type": "Major" if 'core_courses' in item else "Course",
                  "major": item['metadata']['major'],
                  "category": item['metadata']['category'] if item['id'].isdigit() else 'N/A',
                  "prereq": item.get("prerequisite(s)", "N/A")[0],
                  "coreq": item.get("corequisite(s)", "N/A")[0]}
    )
        documents.append(doc)

    return documents

documents = process_data(SJSU_dataset, Gen_Advising_Dataset)
for d in documents:
  if len(d.metadata) > 1:
    p = d.metadata["prereq"]
    co = d.metadata["coreq"]
    prereq = []
    i = 1
    j = 1
    for c in code:
      if c in p:
        d.metadata[f"prereq_{i}"] = c
        i += 1
      if c in co:
        d.metadata[f"coreq_{j}"] = c
        j += 1
    if i == 1:
      d.metadata[f"prereq_{i}"] = "N/A"
    if j == 1:
      d.metadata[f"coreq_{j}"] = "N/A"
    del d.metadata['prereq']
    del d.metadata['coreq']

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
directory = "./vector__store"
vector_store = Chroma(persist_directory=directory, embedding_function=embeddings)
vector_store.get()

prompt = hub.pull("rlm/rag-prompt")

Llama_model = "meta-llama/Llama-3.2-1B-Instruct"

llm = HuggingFaceEndpoint(repo_id=Llama_model,
                            task="text-generation",
                            max_new_tokens=1024,
                            do_sample=False,
                            repetition_penalty=1.03)

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
        k=5,
        filter=filtered
    )
    # Extract source document content
    source_documents = [doc.page_content for doc in retrieved_docs]
    return {"context": retrieved_docs, "source_documents": source_documents}

def generate(state: State) -> State:
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
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
        time.sleep(5)  # Wait before retrying
      else:
        raise e  # Re-raise the error if it's not a 503
  return "Error: The chatbot service is temporarily unavailable. Please try again later."