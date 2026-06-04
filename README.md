# Conversational AI Agent - Adult Income Domain (HW2)

## Στοιχεία Φοιτητή
* **Όνομα:** Χρυσούλα Α. Βαλίνου
* **ΑΜ:** 09325002

---

## 1. System Overview
This project implements an AI Conversational Agent specialized in the Adult Income dataset (1994 US Census) domain. The agent acts as a unified system capable of understanding natural language to:
1. Answer conceptual questions regarding the dataset's demographics and historical context.
2. Perform real-time machine learning inference to predict if an individual's annual income exceeds $50K.
3. Execute mathematical calculations to derive new insights from numerical inputs.

---

## 2. Architecture
The system is built using **LangGraph** for workflow and memory management (via `MemorySaver`) and **FastAPI** for the REST API. The core intelligence is powered by a Cloud LLM (Gemini) acting as a tool-calling ReAct Agent. 

The agent autonomously routes user queries to one of three available tools based on the context:
* **`domain_knowledge_retriever`**: Triggered for theoretical, historical, or terminology-related questions about the dataset.
* **`predict_income`**: Triggered when the user provides individual characteristics (features) and requests an income estimation.
* **`calculator` (Bonus Task 5)**: Triggered for queries requiring precise numerical reasoning or math operations (e.g., converting weekly hours to annual hours).

---

## 3. Knowledge Base
To equip the agent with domain-specific expertise, 5 documents were collected covering:
* The structure, origin, and features of the 1994 US Census Database.
* Demographic disparities of that era.
* Wealth distribution and employment statistics.

**RAG Implementation:** These texts were embedded using the HuggingFace `all-MiniLM-L6-v2` model and persistently stored in a local **ChromaDB** vector store (`data/vector_store`). The agent queries this database to retrieve semantically relevant chunks, allowing it to ground its answers to questions like: *"What are the main features of the dataset?"* or *"How does education impact income according to the data?"*.

---

## 4. HW1 Model Integration
The `predict_income` tool utilizes the **`best_model.pkl`** (Random Forest Classifier) and **`scaler.pkl`** artifacts carried over directly from Homework 1. 

The tool expects a JSON string containing the individual's features as input. It then automatically applies the entire preprocessing pipeline (Encoding, Feature Engineering, Scaling) and returns the final prediction (>50K or <=50K) along with the calculated probability.

**Important Note Regarding Inference Preprocessing:** Based on the grading feedback from HW1, the capping logic (IQR method) was corrected in the current pipeline. Specifically, the `capital_gain` and `capital_loss` columns were excluded from the clipping process to prevent wiping out their valuable information (due to extreme sparsity). While this causes a slight deviation from the strict output of the HW1 `/predict` endpoint, it fixes a critical scientific bug, allowing the model to properly evaluate financial gains during inference.

---

## 5. Example Conversations

### (a) RAG Retrieval Response
```text
User: "What is the dataset about?"

Agent: "Based on the retrieved context, the dataset contains information from the 1994 United States Census Database (Adult Income). It features a mix of continuous and categorical variables regarding demographics and employment to analyze wealth distribution."
```

### (b) Prediction & Calculator Response (Tools execution)
```text
User: "If a 45-year-old married man works in the private sector for 55 hours a week with a capital gain of 15000, what is his likely income? Also, calculate how many hours he works in a 52-week year."

Agent: [Calls predict_income and calculator tools]
"Based on the characteristics provided, the model predicts that this individual earns >50K (Probability: 82.5%). 
Additionally, working 55 hours a week for 52 weeks results in a total of 2860 working hours per year."
```

## 6. Installation & Execution

To run the application locally, execute the following commands in your terminal:

1. **Clone the repository and navigate to the directory:**

   ```bash
   git clone https://github.com/ChrysoulaValinou/hw2-adult-income.git
   cd hw2-adult-income
   ```

2. **Create and activate a virtual environment:**

```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
```

3. **Install the required dependencies**

```bash
pip install -r requirements.txt
```

4. **Set your LLM API Key as an environment variable:**

```bash
set GOOGLE_API_KEY=your_actual_api_key_here
```

5. **Start the FastAPI server:**

```bash
python main.py
```

## 7. Example API Call

Once the server is running, you can interact with the `/chat` endpoint to test the agent. Below are examples using both `curl` and Python `requests`.

**Option A: Using cURL**
```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/chat](http://127.0.0.1:8000/chat)' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "message": "What factors influence income the most in this dataset?",
  "session_id": "test_session_1"
}'
```

**Option B: Using Python requests**

```bash
import requests

url = "[http://127.0.0.1:8000/chat](http://127.0.0.1:8000/chat)"
payload = {
    "message": "What factors influence income the most in this dataset?",
    "session_id": "test_session_1"
}

response = requests.post(url, json=payload)
print(response.json())
```