import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Function to extract text from a PDF
def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save FAISS vector store for a PDF
def create_faiss_index_for_pdf(pdf_path, api_key, output_folder):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_file = os.path.join(output_folder, f"{pdf_name}")
    
    # Read PDF and split into chunks
    text = get_pdf_text(pdf_path)
    text_chunks = get_text_chunks(text)
    
    # Generate embeddings and create vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    # Save the vector store
    vector_store.save_local(output_file)

def main():
    # Define folders
    pdf_folder = "pdfs"
    output_folder = "faiss_index"
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Prompt for API key
    api_key = input("Enter your Google API Key: ").strip()
    
    if not api_key:
        print("API Key is required. Exiting...")
        return
    
    # Iterate through all PDFs in the folder
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file_name)
            print(f"Processing {pdf_path}...")
            try:
                create_faiss_index_for_pdf(pdf_path, api_key, output_folder)
                print(f"Successfully processed and saved FAISS index for {file_name}")
            except Exception as e:
                print(f"Failed to process {file_name}: {e}")
    
    print("Processing complete.")

if __name__ == "__main__":
    main()
