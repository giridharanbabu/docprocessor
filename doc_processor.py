import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_pdf_viewer import pdf_viewer
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
import os, json, requests, tempfile
from datetime import datetime

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "summary_result" not in st.session_state:
    st.session_state.summary_result = None
if "raw_documents" not in st.session_state:
    st.session_state.raw_documents = []
# Initialize session state for file previews
if "file_previews" not in st.session_state:
    st.session_state.file_previews = {}


def read_data(files):
    documents = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
            tmp_file.flush()
        try:
            if file.name.endswith(".pdf"):
                if os.path.getsize(tmp_file_path) == 0:
                    st.error(f"Uploaded file {file.name} is empty.")
                    continue
                pdf_reader = PdfReader(tmp_file_path)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        documents.append(
                            Document(page_content=text, metadata={"source": file.name, "page_number": page_num + 1}))
                    else:
                        st.warning(f"Page {page_num + 1} in {file.name} is empty.")
            elif file.name.endswith(".txt"):
                loader = TextLoader(tmp_file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file.name
                documents.extend(docs)
            elif file.name.endswith(".csv"):
                loader = CSVLoader(tmp_file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file.name
                documents.extend(docs)
            else:
                st.error(f"Unsupported file type: {file.name}")
        except Exception as e:
            st.error(f"Error processing file {file.name}: {e}")
        finally:
            os.remove(tmp_file_path)
    return documents


def summarizer(model_name, _raw_documents):
    from langchain.chains.summarize import load_summarize_chain
    llm = Ollama(model=model_name)
    chain = load_summarize_chain(llm, chain_type="stuff")
    clean_docs = [doc for doc in _raw_documents if isinstance(doc, Document)]
    return chain.invoke(clean_docs)


def query_document(user_question, key):
    system_prompt = st.text_area("System Prompt",
                                 value="You are a helpful assistant. Answer the question as detailed as possible from the provided context. If the answer is not in the provided context say answer is not available in the context", key=f"system_prompt_{datetime.now().timestamp()}")
    if user_question:
        result = get_conversational_chain(user_question, "llama3", system_prompt)
        st.session_state.conversation_history.append({"user": user_question, "response": result['result']})
    st.write("Conversation History")
    for turn in st.session_state.conversation_history:
        st.markdown(f"**User:** {turn['user']}")
        st.markdown(f"**Assistant:** {turn['response']}")


def get_conversational_chain(ques, llm_model, system_prompt):
    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model=llm_model)
    vector_store = FAISS.from_documents(st.session_state.raw_documents, embeddings)
    retriever = vector_store.as_retriever()
    llm = Ollama(model=llm_model, verbose=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                           return_source_documents=True)
    return qa_chain.invoke({"query": ques})


def extract_information_with_ollama(text, template):
    prompt = f"Extract structured information from the text using this template: {json.dumps(template)} Text: {text}"
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 2.5, "top_p": 0.99, "top_k": 100}
    }
    response = requests.post("http://localhost:11434/api/generate", json=payload, stream=False)
    json_data = json.loads(response.text)
    return json.dumps(json.loads(json_data["response"]), indent=2)


st.title("Documents Processor")
tab_h1, tab_h2 = st.tabs(['Text Processor', 'Image Processor'])

with tab_h1:
    col1, col2, col3 = st.columns([3, 6, 3])
    data_files = col1.file_uploader("Upload your Files and Click on Submit to Process", accept_multiple_files=True,
                                    key="text_files_uploader")
    tab1, tab2, tab3, tab4 = col2.tabs(['Summary', 'Conversation History', 'Structure Output', 'Audio Search'])
    if data_files and col1.button("Submit & Process"):
        for file in data_files:
            if file.name not in st.session_state.file_previews:
                with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp_file:
                    tmp_file.write(file.read())
                    tmp_file_path = tmp_file.name
                st.session_state.file_previews[file.name] = tmp_file_path
                st.write(tmp_file_path)

        with st.spinner("Processing..."):
            st.session_state.raw_documents = read_data(data_files)
    with tab1:
        if st.session_state.raw_documents:
            with st.spinner("Processing Summary..."):
                st.session_state.summary_result = summarizer("llama3", st.session_state.raw_documents)
                st.write(st.session_state.summary_result['output_text'])
    with tab2:

        user_question = st.text_input("Ask a Question from the Files")
        with st.spinner("Fetching Conversations..."):
            if user_question:
                query_document(user_question, 'Conversation')
    with tab3:
        if st.session_state.raw_documents:
            input_data = st.text_input("Enter keys to extract")
            if input_data:
                keys = input_data.split(',')
                template = {key: "" for key in keys}
                # if not st.session_state.summary_result:
                #     with st.spinner("Processing Summary first..."):
                #         st.session_state.summary_result = summarizer("llama3", st.session_state.raw_documents)
                with st.spinner("Extracting data..."):
                    structured_output = extract_information_with_ollama(st.session_state.summary_result['output_text'],
                                                                        template)
                    st.json(structured_output)
    with tab4:
        from speechtotext import audio_processing

        text = audio_processing()
        # st.write(text)
        if text:
            text = st.text_area("Question", value=text, key="audio_question")
            query_document(text, 'audio_question')
            # if answer:
            #     text_transcribed = st.text_area("Transcribed Text", value=answer['result'], height=400, disabled=True)
    with col3:
        pdf_viewer("temp.pdf", width=300, height=300)
