import time
import warnings
import json
import os
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


os.environ["HF_TOKEN"] = "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"

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    if not json_data:
        raise ValueError("JSON data is empty")

    print(f"Successfully loaded {len(json_data)} courses")
    return json_data


# Process dataset (Courses & Majors)


def process_data(json_data):
    documents = []

    for item in json_data:
        majors.append(item['metadata']['major']) if item['metadata']['major'] not in majors else None
        if item['id'].isdigit():
            category.append(item['metadata']['category']) if item['metadata']['category'] not in category else None
        title = item.get('title', 'N/A')
        if title != "N/A":
            class_name = title.split("-")[0].strip()
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
        metadata={"title": item.get('title', 'N/A'), "type": "Major" if 'core_courses' in item else "Course", "major": item['metadata']['major'], "category": item['metadata']['category'] if item['id'].isdigit() else 'N/A'}
    )
        documents.append(doc)

    return documents


warnings.filterwarnings('ignore')

# Document Processing Setup
print("Initializing text splitter and embeddings...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", ",", " "]
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Model Initialization(FLAN-T5)
print("Loading FLAN-T5 model and tokenizer...")

# Load base FLAN-T5 model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.clean_up_tokenization_spaces = True
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    device_map="auto",
    # torch_dtype=torch.float32,
)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=1024,
    temperature=0.1, # Lower temperature for factual tasks, higher for creativity
    top_p=0.9,
    do_sample=False,
    repetition_penalty=1.2
)

llm = HuggingFacePipeline(pipeline=pipe)
print("Model loaded successfully!")

prompt = PromptTemplate(
    template="""Use the provided context to answer the question.

    Context: {context}

    Question: {question}
    """,
    input_variables=["context", "question"]
)


# Vector Store Creation
def create_vector_store(documents):
    print("Splitting documents into chunks...")
    # splits = text_splitter.split_documents(documents)
    splits = documents
    print(f"Creating vector store with {len(splits)} chunks...")
    vector_store = FAISS.from_documents(splits, embeddings)
    print("Vector store created successfully!")
    return vector_store


# RAG Chain Setup
def setup_rag_chain(vector_store, search_kwargs):
    # print("Setting up RAG chain...")
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever = vector_store.as_retriever(
            # search_kwargs={"k": 3, "filter": {"major": "Mathematics"}},  # Increase retrieved chunks
            search_kwargs=search_kwargs,
            # search_type="mmr",  # Use Maximal Marginal Relevance for diverse results
            # search_kwargs={"fetch_k": 20, "lambda_mult": 0.7}  # Fetch 20 and re-rank top 8
        ),
        chain_type_kwargs={
            "prompt": prompt
            # "verbose": True
        },
        return_source_documents=True
    )
    return chain

def query_visa_assistant(qa_chain, question):
    result = qa_chain.invoke({"query": question})

    # # Print retrieved documents for debugging
    # for i, doc in enumerate(result.get("source_documents", [])):
    #     print(f"Source {i+1}: {doc.page_content[:300]}...\n")

    return {
        "answer": result.get("result", "No response generated"),
        "source_documents": result.get("source_documents", [])
    }

def initialize_qa_system(json_data):
    print("Processing data and creating vector store...")
    documents = process_data(json_data)
    vector_store = create_vector_store(documents)
    # qa_chain = setup_rag_chain(vector_store)
    print("QA system initialized successfully!")
    return vector_store

def collect_model_responses(vector_store, questions):
    data = []

    # Process each question
    for i, question in enumerate(questions, 1):
        search_kwargs = {"k": 3}
        formatted_question = question
        for k,v in class_mapping.items():
            if k in question and v not in question:
                formatted_question = question.replace(k, v)
        for c in category:
            if c in question:
                search_kwargs["filter"] = {"category": c}
        for m in majors:
            if m in question:
                search_kwargs["filter"] = {"major": m}
        print(f"Processing question {i}/{len(questions)}")
        start_time = time.time()
        print(search_kwargs)
        qa_chain = setup_rag_chain(vector_store, search_kwargs)
        response = query_visa_assistant(qa_chain, formatted_question)
        sources = []
        for doc in response['source_documents']:
            sources.append(doc.metadata['title'])
        response_time = time.time() - start_time

        data.append({
            'question_id': i,
            'question': question,
            'response': response['answer'],
            'documents': sources,
            'response_time': round(response_time, 2),
            'response_length': len(response['answer']),
            'source_documents': response['source_documents']
        })

    df = pd.DataFrame(data)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)

    return df

class_mapping = {}
majors = []
category = []


# dataset_path = "/content/drive/MyDrive/CMPE-295A/dataset/SJSU_courses_majors_dataset.json"
dataset_path = "SJSU_courses_with_metadata_updated.json"
# gen_path = "complete_Gen_Advising.json"
SJSU_dataset = load_json_data(dataset_path)
# Gen_Advising_Dataset = load_json_data(gen_path)

vector_store = initialize_qa_system(SJSU_dataset)
majors = majors[-3:]
def get_chatbot_response(user_question):
    search_kwargs = {"k": 3}
    qa_chain = setup_rag_chain(vector_store, search_kwargs)
    response = qa_chain.invoke({"query": user_question})
    return response.get("result", "No response generated")
if __name__ == "__main__":
    print("LLM.py is now a module and should be imported into Flask.")
