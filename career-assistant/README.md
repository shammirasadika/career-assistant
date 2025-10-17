# **Career Assistant – Question & Answer System**

---

## Scenario Selected
This project implements a **Career Assistant QA System** designed to help users explore occupations, duties, required skills, and salary ranges in Australia.  
It uses both **structured data (salary lookup tool)** and **unstructured text (OSCA document)** to provide reliable, context-aware answers.

**Why this scenario was selected:**
- It represents a realistic and socially useful application of AI.  
- It demonstrates how RAG (Retrieval-Augmented Generation) and tool integration can complement each other.  
- It allows testing of routing, reasoning, and information retrieval in a single workflow.

## Target Users
- University students exploring career paths  
- Recent graduates seeking role expectations and pay ranges  
- Professionals planning reskilling or job transition  

---


## Documents and Tools Used

### Grounding Document (for RAG)
- **OSCA – Occupation Standard Classification for Australia, 2024, Version 1.0 (ABS)**  
  Used as the **knowledge base** for retrieving occupational definitions, duties, and skill sets.  
  Relevant sections were embedded and indexed for retrieval within the RAG pipeline.

### Tool
- **Salary Lookup Tool** – A CSV-based dataset mapping *Occupation × State → Average Salary*.  
  Automatically triggered for salary-related queries (e.g., “salary,” “wage,” “income”).

- **Prompt-driven routing logic** to decide between RAG, tool calling, or both

- **Local Gradio interface** for querying

---

##  LLMs and Frameworks Adopted
- **Primary LLM:** OpenAI ChatGPT-based model  
- **Fallback:** Groq API (used if primary model fails or times out)  
- **Frameworks / Libraries:**  
  - *Gradio* – interactive web interface  
  - *SentenceTransformers / OpenAI Embeddings* – for semantic similarity and vector search  

---

 ## Routing Logic Overview
Routing determines whether a query follows the **RAG path**, **Tool path**, or **Both**, based on the detected query type.

|-----------------------------------------------------------------------------------------------------------------|
|        Query Type            |      Route    |                      Example                                     |
|:-----------------------------|:--------------|:-----------------------------------------------------------------|
| Skills / Duties / Definitions| **RAG Path**  | “What skills are needed for a Project Manager?”                  |
| Salary / Pay / Wage          | **Tool Path** | “What is the average salary for a Data Analyst in NSW?”          |
| Combined (skills + salary)   | **Both Paths**|“What are the Duties and average salary of a Software  Engineer?” |
|-----------------------------------------------------------------------------------------------------------------|

**Process Flow**
1. User submits query via Gradio UI.  
2. Controller (`handle_query`) checks keywords and confidence thresholds.  
3. `route_query()` decides whether to use RAG, Tool, or Both.  
4. RAG retrieves data from OSCA; Tool fetches from the salary CSV.  
5. The LLM merges outputs and formats a final answer with citations.  
6. Post-processing removes duplicates and numbers references correctly.

---

## Project Structure
```
career-assistant/
│
├── app.py
├── config.yml
├── prompts.py
├── rag.py
├── llm.py
├── data_prep.py
├── tools/
│   └── salary_tool.py
├── data/
├── evaluation/
├── requirements.txt
└── README.md
```

---


## How to Run the System Locally

Follow the steps below to run the **Career Assistant** system on your local machine using **VS Code Terminal**.

### Step 1: Open the project in VS Code
1. Open **VS Code**.  
2. Go to **File → Open Folder →** select your project folder named `career-assistant`.  
   Make sure this folder contains:
   - `app.py`, `rag.py`, `llm.py`, `prompts.py`, `requirements.txt`
   - A subfolder called `/data` with:
     - `OSCA - Occupation Standard Classification for Australia, 2024.pdf`
     - `salary_data.csv`

### Step 2: Open Terminal
In VS Code, open the integrated terminal:
- go to **View → Terminal**


### Step 3: Set Up a Virtual Environment
   - Open a terminal in VS Code.
   - Run the following commands:
     ```bash
     python -m venv .venv
     ```
   - Then activate it:
     - On **Windows**:
       ```bash
       .venv\Scripts\activate
       ```
     - On **Mac/Linux**:
       ```bash
       source .venv/bin/activate
       ```

### Step 4:Install all dependencies listed in requirements.txt
- Run the command below to install all required libraries:
  ```bash
   pip install -r requirements.txt
   ```

### Step 5: Run the Application
 ```bash
python app.py
```
You’ll see output like: Running on local URL: http://127.0.0.1:7860

### Step 6: Open the Web Interface
Copy the URL shown in the terminal (e.g. http://127.0.0.1:7860)
and paste it into your web browser (Chrome/Edge).
You’ll see the Career Assistant Chat Interface appear.

### Step 7: Ask Questions
You can now test the chatbot. Examples:

- What skills are needed for a Software Engineer?
- What is the average salary for a Data Analyst in NSW?
- What are the duties and salary of a Project Manager?

The system will automatically:
- Use RAG path for text-based questions.
- Use Tool path for salary queries.
- Use Both paths for mixed questions.

### Step 8: Stop the Application
- To stop the server, press `Ctrl + C` in the terminal.

---

## Example User Queries and Expected Outputs

Example 1:
Q: What are the duties of a software engineer?
→ RAG Path → Answer extracted from OSCA (Skill Level 1 occupations, ICT category).
Expected Output: “Software Engineers design, develop, test, and maintain software systems... [Source: OSCA p. 234]”

Example 2:
Q: What is the average salary of a Project Manager in Victoria?
→ Tool Path → Retrieved from Salary CSV.
Expected Output: “Average salary for Project Manager in VIC is approximately $126,000 per year.”

---


## Limitations and Challenges

1. Data Coverage
Some specialised or new job roles (e.g., AI Engineer, Cloud Security Specialist) are not fully represented in the OSCA document, resulting in occasional incomplete or low-confidence retrievals.

2. Salary Dataset Scope
The salary tool only covers major Australian states (NSW, VIC, QLD, SA, WA), so queries about other territories may return “data not available.”

3. Routing Confusion
Similar job titles (like Developer, Programmer, and Software Engineer) can confuse the routing logic and sometimes trigger the wrong processing path.

4. Context and Token Limit
Long or multi-part queries may exceed the model’s context window, causing partial or truncated answers from the RAG retriever.

5. API and Mixed Query Handling
Although the Groq fallback adds reliability, occasional network or timeout issues remain.  
Hybrid queries combining skills and salary sometimes generate slightly unbalanced responses.

Overall, the system performs well but could improve with expanded datasets, better synonym mapping, and enhanced context handling.

---


## Summary
The Career Assistant system combines document retrieval and data-driven reasoning to deliver clear, relevant answers about occupations, skills, and salaries.  
It integrates a RAG-based approach with a Salary Lookup Tool, allowing the model to use both unstructured text and structured numeric data.  
This hybrid design ensures that users receive accurate and context-aware responses whether they ask about job descriptions, duties, or salary ranges.  

By grounding answers in the official OSCA document and combining them with factual salary data, the system provides transparency and reliability—two qualities essential in career-related decision support.  
Overall, the Career Assistant demonstrates how retrieval-augmented AI can connect real information with reasoning to create a practical and explainable solution for everyday users.

---