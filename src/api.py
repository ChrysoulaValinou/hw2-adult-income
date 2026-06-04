from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.agent import agent_executor

# Δημιουργούμε το router για το endpoint
router = APIRouter()

# Ορίζουμε το σχήμα εισόδου (τι περιμένουμε να μας στείλει ο χρήστης) σύμφωνα με την εκφώνηση
class ChatRequest(BaseModel):
    message: str
    session_id: str

# Ορίζουμε το σχήμα εξόδου (τι θα επιστρέψουμε)
class ChatResponse(BaseModel):
    response: str

# --------------------------------------------------------
# TASK 4: Το κανονικό endpoint (Το αφήνουμε όπως ήταν)
# --------------------------------------------------------
@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # Ετοιμάζουμε το configuration για τη μνήμη, χρησιμοποιώντας το session_id του χρήστη
    config = {"configurable": {"thread_id": request.session_id}}
    
    # Καλούμε τον agent που φτιάξαμε στο Task 3
    result = agent_executor.invoke(
        {"messages": [("user", request.message)]},
        config=config
    )
    
    # Εξάγουμε το τελικό κείμενο
    final_content = result['messages'][-1].content
    if isinstance(final_content, list) and len(final_content) > 0:
        clean_answer = final_content[0].get('text', str(final_content))
    else:
        clean_answer = final_content
        
    return ChatResponse(response=clean_answer)

# --------------------------------------------------------
# TASK 6 (BONUS): Το νέο Streaming endpoint
# --------------------------------------------------------
@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    def event_generator():
        config = {"configurable": {"thread_id": request.session_id}}
        
        # Το stream_mode="messages" μας δίνει την απάντηση λέξη-λέξη
        for chunk, metadata in agent_executor.stream(
            {"messages": [("user", request.message)]}, 
            config=config,
            stream_mode="messages"
        ):
            # Ελέγχουμε αν το κομμάτι (chunk) έχει κείμενο
            if chunk.content and isinstance(chunk.content, str):
                # Το SSE (Server-Sent Events) απαιτεί το format: data: <κείμενο>\n\n
                yield f"data: {chunk.content}\n\n"

    # Επιστρέφουμε την απάντηση ως συνεχόμενη ροή (stream)
    return StreamingResponse(event_generator(), media_type="text/event-stream")