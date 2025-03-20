import re
import os
import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv
import random

def get_clean_text(text):

    """Cleans text by removing unwanted characters and excessive whitespace."""

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,;!?-]', '', text)
    return text.strip()

def prepare_documents(devices_list):

    '''This function takes a list of device urls and returns a list of Document objects'''

    devices_list=['page_source_1.html','page_source_2.html','page_source_3.html','page_source_4.html']
    final_document_objects=[]
    output_dir = './Winng_design_diagrams'
    if os.path.exists(output_dir):
        # Remove the existing directory and its contents
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        #os.rmdir(output_dir)
    else:
        os.makedirs(output_dir)
    
    for idx, device_url in tqdm(enumerate(devices_list)):

        image_file_path=None
        user_agents = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/537.36"
        ]
        headers = {"User-Agent": random.choice(user_agents)}
        response = requests.get(device_url, headers=headers)
        print(response.status_code)
        time.sleep(10)
        soup = BeautifulSoup(response.content, "html.parser")
        des=soup.find(class_="description")
        #img=soup.find_all("svg")
        
        # Extract the description (all <p> tags before the benefits)
        description = " ".join(p.get_text() for p in des.find_all("p")[:-1])
        cleaned_description=get_clean_text(description)
        #print("Description:")
        #print(description)

        # Extract the system benefits (all <li> tags within the <ul>)
        benefits = [li.get_text() for li in des.find_all("li")]
        #print("\nSystem Benefits:")
        benefits_string=''
        for benefit in benefits:
            benefits_string=benefits_string+benefit+'.'

        cleaned_benefits=get_clean_text(benefits_string)
        content_string=cleaned_description+' '+cleaned_benefits

        svg_img=soup.find_all("svg")
        for i in svg_img:
            if('''color-interpolation-filters="sRGB"''' in str(i)):
                #print(i)
                #print('-'*50)
                image_file_path=os.path.join(output_dir,"block_diagram_{}.svg".format(idx))
                with open(image_file_path, "w", encoding="utf-8") as f:
                    f.write(str(i))
                #print("SVG block diagram saved successfully!")
        meta_data_dict={'image_path':image_file_path,'page_url':device_url}
        final_document_objects.append(Document(page_content=content_string,metadata=meta_data_dict))
    return final_document_objects

def crawl_website():

    '''This function crawls the Renesas website and extracts the device descriptions and block diagrams'''

    user_agents = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/537.36"
    ]
    electronics = ['computing','home-theater-entertainment','power-adapters-chargers','wearables']
    devices_list = []
    for electronic in electronics:
        base_url='https://www.renesas.com/en/applications/consumer-electronics/{}'.format(electronic)
        headers = {"User-Agent": random.choice(user_agents)}
        response = requests.get(base_url, headers=headers)
        print(response.status_code)
        #wait sometime before the next request
        time.sleep(2)
        #print(response.text)
        soup = BeautifulSoup(response.content, "html.parser")

        app_list = soup.find('div', class_='application-category-list')

        if app_list:
            urls = [a['href'] for a in app_list.find_all('a', href=True)]

            for idx in range(0, len(urls)):    
                if '#' in urls:
                    urls.remove('#')
                else:
                    break
            for url in urls:
                #print(url)
                key_word = url.replace('/en/applications/consumer-electronics/{}'.format(electronic), '')
                #print(key_word)

                devices_list.append(base_url + key_word)

    return devices_list


def get_embeddings():

    '''We are using the HuggingFace BAAI/bge-large-en model to get the embeddings for the documents'''
    
    Bge_model_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en", encode_kwargs={'normalize_embeddings': True})
    return Bge_model_embeddings

def faiss_store(documents, embedding_model):
    vector_store = FAISS.from_documents(tqdm(documents), embedding_model)
    return vector_store

def get_relavant_devices(user_input, vectorstore):
    
    '''This function retrieves the relevant devices based on the user input'''
    
    relavant_documents=[]
    documents_metadata=[]
    #retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    retrieved_documents = vectorstore.similarity_search(query=user_input,k=2)
    relavant_documents.extend(doc.page_content for doc in retrieved_documents)
    documents_metadata.extend(doc.metadata for doc in retrieved_documents)
    return relavant_documents, documents_metadata
    
def llm_chain():

    '''This function sets up the LLM chain'''

    prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Use the following context to answer the user's question.
    Context:
    {context}

    Question:
    {question}

    Answer:""")
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo", temperature=0)
    return LLMChain(llm=llm, prompt=prompt_template)


def initialize_session_state_variables():

    if "contents" not in st.session_state:
        st.session_state.contents = []
    if "selected_content" not in st.session_state:
        st.session_state.selected_content = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_chat" not in st.session_state:
        st.session_state.show_chat = False
    if "crawl" not in st.session_state:
        st.session_state.crawl=crawl_website()
    if "documents" not in st.session_state:
        st.session_state.documents = prepare_documents(st.session_state.crawl)
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = get_embeddings()
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = faiss_store(st.session_state.documents, st.session_state.embedding_model)
    if "llm_chain" not in st.session_state:
        st.session_state.llm_chain = llm_chain()
    if "top_2_devices" not in st.session_state:
        st.session_state.top_2_devices=None
    if "top_2_metadata" not in st.session_state:
        st.session_state.top_2_metadata=None

def user_interface():
    load_dotenv()

    initialize_session_state_variables()

    st.title("AI Powered Renesas electronics device designes finder")

    # Search Interface
    user_input = st.text_input("Search for design:")

    if st.button("Search"):
        # Example content retrieval (Replace with actual FAISS retrieval)
        if user_input.strip():
            st.session_state.contents=[]
            st.session_state.top_2_devices, st.session_state.top_2_metadata = get_relavant_devices(user_input, st.session_state.vector_store)

            for device in st.session_state.top_2_devices:
                st.session_state.contents.append(device)
            st.session_state.selected_content = None  # Reset content selection
            st.session_state.show_chat = False  # Reset chat visibility
            st.session_state.chat_history = []  # Clear chat history
            st.session_state.image = None
            st.session_state.img_count = 0

    # Show search results if available
    if st.session_state.top_2_devices:
        col1, col2 = st.columns(2)
        print('-'*80)
        print('sess_Content : \n {}'.format(st.session_state.contents[0]))
        print('-'*80)
        with col1:
            st.subheader("Top-1 Document")
            st.text_area("Content Description", st.session_state.contents[0], height=150, key="content1")
            if st.button("Chat with me", key="chat1"):
                st.session_state.image =st.session_state.top_2_metadata[0]['image_path']
                st.session_state.img_count=1
                st.session_state.chat_history = []
                st.session_state.selected_content = 'Top-1 Document'
                st.session_state.show_chat = True

        with col2:
            st.subheader("Top-2 Document")
            st.text_area("Content Description", st.session_state.contents[1], height=150, key="content2")
            if st.button("Chat with me", key="chat2"):
                st.session_state.image=st.session_state.top_2_metadata[1]['image_path']
                st.session_state.img_count=1
                st.session_state.chat_history = []
                st.session_state.selected_content = 'Top-2 Document'
                st.session_state.show_chat = True
        if st.session_state.img_count==1:
            col_center = st.columns([(10 - 9) / 2, 9, (10 - 9) / 2])[1]
            with col_center:
                st.image(st.session_state.image, caption="Block Diagram", width=900)

    # Chatbot Interface
    if st.session_state.show_chat and st.session_state.selected_content:
        st.subheader("Chatbot Interface for: {}".format(st.session_state.selected_content))

        user_message = st.chat_input("Enter your question:")
        if user_message:
            if st.session_state.selected_content == 'Top-1 Document':
                llm_model=llm_chain()
                bot_response = llm_model.run(context=st.session_state.contents[0], question=user_message)
                st.session_state.chat_history.append({"user": user_message, "bot": bot_response})
            else:
                llm_model=llm_chain()
                bot_response = llm_model.run(context=st.session_state.contents[1], question=user_message)
                st.session_state.chat_history.append({"user": user_message, "bot": bot_response})

        # Display chat history
        for chat in st.session_state.chat_history:
            st.chat_message("user").markdown(chat["user"])
            st.chat_message("assistant").markdown(chat["bot"])

if __name__=="__main__":

    user_interface()