import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import pandas as pd
import re
from langchain.embeddings import HuggingFaceEmbeddings
from nltk.stem import WordNetLemmatizer
import nltk
import time
from textblob import TextBlob

llm = OllamaLLM(model="llama3.2:3b", temperature=0.6)
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')

def create_vectordb_faiss():
    start_time = time.time()
    df = pd.read_csv('dataset/cleaned_product_data_v2.csv')
    raw_documents = [
        Document(
            page_content=str(row['combined_text']),
            metadata={'product_name': row['product_name'], 'product_id': row['product_id'],'product_link': row['product_link'] }
        )
        for _, row in df.iterrows() if pd.notna(row['combined_text'])
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    documents = text_splitter.split_documents(raw_documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = FAISS.from_documents(documents, embeddings)
    db.save_local("all-mpnet-base-v2_faiss")
    end_time = time.time()
    duration = end_time - start_time
    print(f"Time taken for creating vectordb embeddings: {duration:.2f} seconds\n")
    return db

def get_recommendations(vectorstore, user_input, top_k=3):
    results = vectorstore.similarity_search_with_score(user_input, k=top_k * 5)
    seen_products = {}
    
    for result, score in results:
        name = result.metadata['product_name']
        link = result.metadata['product_link']
        id = result.metadata['product_id']
        if name not in seen_products:
            seen_products[name] = {
                'chunks': [], 
                'score': score,
                'id': id,       
                'link': link
            }
        seen_products[name]['chunks'].append(result.page_content)

    final_results = []

    for name, data in seen_products.items():
        final_results.append({
            'product_id': data['id'],
            'product_name': name,
            'combined_content': " ".join(data['chunks']),
            'score': data['score'],
            'product_link': data['link']
        })

    return final_results[:top_k]

def preprocess_user_input(user_input):
    user_input = user_input.lower()
    user_input = re.sub(r'[^\w\s]+', ' ', user_input).strip()
    corrected_input = str(TextBlob(user_input).correct())
    tokens = corrected_input.split()
    lemmatized_input = ' '.join([lemmatizer.lemmatize(token) for token in tokens])
    return lemmatized_input

def create_llm_prompt(user_query, product_candidates):
    responses = []
    prompt = f"User query: '{user_query}'\n"
    prompt += (
        "\nFor the following product, provide only the following information:\n"
        "'Explanation: [One-line reason why this product is a good fit for the user's needs, focusing on its key features and how they align with the User query]'\n"
        "If it doesn't match directly, highlight product's most useful features."
        "Do not include any extra information or overall summary.\n"
    )
    for i, product in enumerate(product_candidates):
        product_link = product['product_link']
        product_id = product['product_id']
        
        product_prompt =prompt + f"Product Name: {product['product_name']}, Description: {product['combined_content']}\n"
        response = llm_refine_recommendations(product_prompt)
        responses.append({
            'product_id': product_id,
            'product_name':product['product_name'],
            'product_link': product_link,
            'response': response
        })
    return responses 

def llm_refine_recommendations(prompt):
    print("Sending prompt to the local LLM")
    response =llm.invoke(prompt)
    return response

def get_llm_recommendations(vectorstore, user_query):
    start_time = time.time()
    product_candidates = get_recommendations(vectorstore, user_query)
    llm_recommendations = create_llm_prompt(user_query, product_candidates)
    end_time = time.time()
    print(f'Duration {end_time - start_time}')
    return llm_recommendations

def main():
    st.title("AI Matchmaker Test")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    try:
        vector_db = FAISS.load_local("all-mpnet-base-v2_faiss", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading vector database: {e}")
        return
    try:
        vector_db = FAISS.load_local("all-mpnet-base-v2_faiss", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading vector database: {e}")
        return

    if 'user_query' not in st.session_state:
        st.session_state.user_query = ""
    if 'refined_query' not in st.session_state:
        st.session_state.refined_query = None
    if 'feedback' not in st.session_state:
        st.session_state.feedback = None

    st.session_state.user_query = st.text_area("Let PieceX match you with the ideal source code solution. Input your requirements:",
                                               value=st.session_state.user_query,
                                               key='user_query_input')

    if st.session_state.refined_query:
        user_query = preprocess_user_input(st.session_state.refined_query)
    else:
        user_query = preprocess_user_input(st.session_state.user_query)

    if st.session_state.user_query or st.session_state.refined_query:
        print(f'User query: {st.session_state.user_query}')
        recommendations = get_llm_recommendations(vector_db, user_query)
        key = 0
        while True:
            key += 1
            if recommendations:
                st.write("Here are some product recommendations:")
                for product in recommendations:
                    st.markdown(f"**Product Name:** {product['product_name']}")
                    print(product['response'])
                    st.markdown(product['response'])
                    st.markdown(f"[View Product]({product['product_link']})")

                st.session_state.feedback = st.radio("Did you find what you were looking for?", ["Yes", "No"], index=0, key=str(key) + 'loop_key')
                if st.session_state.feedback == "No":
                    additional_details = st.text_area("Please provide more details to refine your search:", key=str(key) + 'query_key')
                    if additional_details:
                        if st.session_state.refined_query:
                            st.session_state.refined_query += " " + additional_details
                        else:
                            st.session_state.refined_query = additional_details

                        print(f"Updated refined query: {st.session_state.refined_query}")
                        refined_query = preprocess_user_input(st.session_state.refined_query)
                        recommendations = get_llm_recommendations(vector_db, refined_query)
                    else:
                        break
                else:
                    break
            else:
                st.write("Sorry, I couldn't find any recommendations based on your query.")
                break


if __name__ == "__main__":
    main()
