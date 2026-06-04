import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Ορίζουμε τους φακέλους μας
DOCUMENTS_DIR = "data/documents"
VECTOR_STORE_DIR = "data/vector_store"

def get_vector_store():
    """
    Ελέγχει αν υπάρχει ήδη η βάση (vector store). Αν υπάρχει, τη φορτώνει.
    Αν δεν υπάρχει, διαβάζει τα αρχεία .txt, τα κόβει σε κομμάτια, και τη δημιουργεί!
    """
    # Χρησιμοποιούμε ένα δωρεάν, τοπικό embedding model από το HuggingFace
    # Αυτό μετατρέπει το κείμενο σε αριθμούς (διανύσματα)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Αν η βάση υπάρχει ήδη στον δίσκο, την φορτώνουμε απλά (για ταχύτητα!)
    if os.path.exists(VECTOR_STORE_DIR) and os.listdir(VECTOR_STORE_DIR):
        print("Φόρτωση υπάρχοντος Vector Store από τον δίσκο...")
        vector_store = Chroma(
            persist_directory=VECTOR_STORE_DIR,
            embedding_function=embeddings
        )
        return vector_store
    
    print("Το Vector Store δεν βρέθηκε. Δημιουργία από την αρχή...")
    
    # 1. Φόρτωση όλων των .txt αρχείων από τον φάκελο data/documents
    loader = DirectoryLoader(DOCUMENTS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Φορτώθηκαν {len(documents)} έγγραφα.")

    # 2. Κόβουμε τα κείμενα σε μικρότερα κομμάτια (chunks)
    # chunk_size: 500 χαρακτήρες ανά κομμάτι
    # chunk_overlap: 50 χαρακτήρες επικάλυψη ώστε να μην χάνεται το νόημα στις ενώσεις
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"Τα έγγραφα κόπηκαν σε {len(chunks)} μικρότερα chunks.")

    # 3. Δημιουργία της βάσης ChromaDB και αποθήκευση στον δίσκο (persist)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR
    )
    print(f"Το Vector Store δημιουργήθηκε και αποθηκεύτηκε στο {VECTOR_STORE_DIR}")
    
    return vector_store

def retrieve_context(query: str, k: int = 3) -> str:
    """
    (Task 1.3) Η συνάρτηση που θα καλεί ο Agent.
    Παίρνει την ερώτηση του χρήστη και επιστρέφει τα 'k' πιο σχετικά chunks 
    ενωμένα σε ένα ενιαίο κείμενο.
    """
    vector_store = get_vector_store()
    
    # Ζητάμε από το Chroma να βρει τα πιο σχετικά κομμάτια
    docs = vector_store.similarity_search(query, k=k)
    
    # Ενώνουμε τα κείμενα των chunks που βρήκαμε
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

# Ένα μικρό τεστ για να δούμε αν δουλεύει!
if __name__ == "__main__":
    print("Δοκιμή του RAG συστήματος...")
    # Την πρώτη φορά που θα τρέξει, θα δημιουργήσει τον φάκελο vector_store!
    test_query = "What happens to income when someone works more than 50 hours?"
    result = retrieve_context(test_query)
    print("\n--- Αποτέλεσμα Αναζήτησης ---")
    print(result)