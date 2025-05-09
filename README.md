# Counsel.NLP

**Smart Academic Advisor**  
CMPE 295A/B Project – San José State University  
Developed by: Jeffrey Ong, Baljot Singh, Pranay Mantramurti  
Advisor: Dr. Magdalini Eirinaki

---

## Introduction

**Counsel.NLP** is a modular academic advising chatbot designed to assist students from the College of Engineering at San José State University. It integrates **Retrieval-Augmented Generation (RAG)**, **vector-based document search**, and **large language models (LLMs)** through a lightweight web application interface. The goal is to deliver **context-aware**, **relevant**, and **accurate academic responses**, while scaling support for student advising.

---

## Dataset

The dataset was created in JSON format and can be found under the [`academic_advising_dataset`](./academic_advising_dataset) folder in this repository and includes:

- **1048 courses** from the official SJSU course catalog  
  - Includes details such as course titles, descriptions, prerequisites, corequisites, and class structures  
- Covers **3 graduate programs**:
  - MS in Artificial Intelligence (MSAI)
  - MS in Computer Engineering (MSCMPE)
  - MS in Software Engineering (MSSE)
- **106 general advising topics** sourced from official SJSU advising and department websites

---

## Preprocessing & Data Collection

To create the dataset:
1. **Course Data** was scraped and normalized from the official SJSU course catalog.
2. **Advising Topics** were collected from SJSU's graduate program advising pages and manually curated.
3. Data was stored in a structured JSON format for efficient retrieval and LLM context integration.

---

## Testbed Setup

We manually crafted **61 evaluation questions** representing real-world advising needs. Each was tested against our chatbot for:
- **BERTScore F1**
- **Answer Presence**
- **Answer Consistency**
- **Response Time** and **Length**

Evaluation results help quantify the chatbot’s reliability and relevance in academic advising contexts.

---

## Model Used

Counsel.NLP uses a **Retrieval-Augmented Generation (RAG)** pipeline with:
- **HuggingFace Embeddings** for vector storage
- **Chroma** as the vector database
- **NVIDIA’s Chat LLM (Llama-3.2-3B-Instruct)** for text generation
- Lightweight orchestration using **LangGraph**

---

## Web Application

The chatbot interface was developed with **Streamlit**, supporting:
- Natural language queries about courses or advising topics
- Upload-based transcript parsing for **personalized course recommendations**
- Real-time response generation and retrieval explanation

You can try the chatbot here 👉 [Counsel.NLP Chatbot](https://counsel-nlp.streamlit.app/)

![chatbot](https://github.com/user-attachments/assets/c0d2b03b-a70d-4839-bcc3-70dfe579cc98)

## Running the Evaluation

To evaluate the chatbot's performance:

1. Open the `Llama_RAG.ipynb` notebook in **Google Colab** using this link:  
   [Run Llama_RAG.ipynb in Google Colab](./Llama_RAG.ipynb)

2. After opening it, click **"Open in Colab"** at the top of the GitHub viewer (if available), or upload it manually to [Google Colab](https://colab.research.google.com/).

3. Once in Colab, **run all cells from top to bottom**.
   
   This will:
   - Load the evaluation dataset
   - Perform document retrieval using the vector store
   - Generate responses with the LLM
   - Compute evaluation metrics like **BERTScore**, **Answer Presence**, and **Consistency**

>  All necessary files like `vector__store.zip` and the `academic_advising_dataset` folder should be obtained from this GitHub repository and manually uploaded to your Google Colab session:
> - [`vector_store.zip`](./vector_store.zip)
> - [`academic_advising_dataset/`](./academic_advising_dataset)
>
> After uploading, make sure to unzip `vector__store.zip` inside Colab before running the retrieval and generation pipeline.

---    
