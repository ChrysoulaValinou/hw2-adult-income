import os
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver  # <-- ΝΕΟ: Εισαγωγή Μνήμης
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI 

# 1. Εισαγωγή των εργαλείων μας
from src.rag import retrieve_context
from src.tools import predict_income, calculator

api_key = os.environ.get("GOOGLE_API_KEY") 

# 2. RAG Tool
@tool
def domain_knowledge_retriever(query: str) -> str:
    """
    Calls the RAG retrieval function. 
    Use this tool ONLY when the user asks a factual, conceptual, or general 
    knowledge question about the domain (e.g., income, demographics, capital gains, dataset context).
    """
    return retrieve_context(query)

# 3. Ορίζουμε το LLM (Το "μυαλό")
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

# 4. Η λίστα με τα εργαλεία
tools = [domain_knowledge_retriever, predict_income, calculator]

# 5. ΠΡΟΣΘΗΚΗ ΜΝΗΜΗΣ (Task 3.2)
# Φτιάχνουμε ένα αντικείμενο μνήμης που θα κρατάει το ιστορικό
memory = MemorySaver()

# 6. Δημιουργία του Agent με τον μηχανισμό μνήμης (checkpointer)
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

# --- Τεστ του Agent με Μνήμη ---
if __name__ == "__main__":
    print("🤖 Ο AI Agent (πλέον με ΜΝΗΜΗ) φορτώθηκε! (Γράψε 'exit' για έξοδο)")
    
    # Το LangGraph χρειάζεται ένα "αναγνωριστικό συζήτησης" (thread_id) 
    # για να ξέρει σε ποιο φάκελο μνήμης να κοιτάξει.
    config = {"configurable": {"thread_id": "session_1"}}
    
    while True:
        user_input = input("\nΕσύ: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Τερματισμός...")
            break
            
        try:
            print("Ο Agent σκέφτεται...")
            
            # Περνάμε το config μαζί με το μήνυμα για να ενεργοποιηθεί η μνήμη
            response = agent_executor.invoke(
                {"messages": [("user", user_input)]}, 
                config=config
            )
            
            final_content = response['messages'][-1].content
            
            if isinstance(final_content, list) and len(final_content) > 0:
                clean_answer = final_content[0].get('text', str(final_content))
            else:
                clean_answer = final_content
                
            print(f"\nAgent: {clean_answer}")
            
        except Exception as e:
            print(f"\n[Σφάλμα]: {e}")