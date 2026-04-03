import os
import json
import gradio as gr
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# Load environment variables (Local use). In HF Spaces, use Secrets!
load_dotenv()

# Environment Variables setup
# Ensure you configure 'PINECONE_API_KEY' and 'HUGGINGFACE_API_TOKEN' in HF Spaces settings.
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# 1. Setup the "Brain"
print("Setting up LLM Endpoint...")
qwen_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.1, # Keep low for consistent JSON routing
    max_new_tokens=512
)

llm = ChatHuggingFace(llm=qwen_endpoint)

# 2. Setup Vector Database Connection
print("Setting up Pinecone VectorStore Connection...")
embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-m3",
    huggingfacehub_api_token=HF_TOKEN
)

index_name = "updcoursessyllabindex" 
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name, 
    embedding=embeddings
)

# 3. The Router Prompt & Chain
router_prompt = PromptTemplate.from_template(
    """You are an assistant for the UP Diliman Math Department.
    
    Chat History:
    {history}
    
    Analyze the user's latest question: "{question}"
    
    You must respond ONLY with a JSON object in the following format:
    {{
        "is_math_course_related": boolean,
        "is_broad": boolean,
        "search_query": "A rewritten standalone version of the user's question optimized for vector search, resolving any pronouns or references from the chat history. Null if is_math_course_related is false.",
        "reason": "A brief explanation of why you categorized it this way"
    }}
    
    Guidelines for 'is_broad':
    - Set to true if the query is asking for a general overview, list of classes, professor summary, or multiple courses.
    - Set to false if the query is asking about specific details (like prerequisites, specific schedule, or a rule from a syllabus).
    
    JSON:"""
)

router_chain = router_prompt | llm

# 4. RAG Prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Academic Advisor for the UP Diliman Institute of Mathematics. 
    Use the following pieces of retrieved context from official course syllabi and the chat history to answer the student's question.
    
    Rules:
    1. If the answer isn't in the context, say you don't know. Do not make up course details.
    2. Be specific about course codes (e.g., Math 21, Math 122).
    3. If prerequisites are mentioned, list them clearly.
    
    Context:
    {context}
    
    Chat History:
    {history}"""),
    ("human", "{question}"),
])

# 5. Core Query Handling Logic
def handle_query(user_input, history=None):
    if history is None:
        history = []
        
    # Format history into a string
    formatted_history = ""
    for p in history:
        formatted_history += f"User: {p[0]}\nAdvisor: {p[1]}\n"
    if not formatted_history:
        formatted_history = "No previous history."
        
    # Step 1: Run the Router
    response = router_chain.invoke({"question": user_input, "history": formatted_history})
    raw_output = response.content
    
    clean_json = raw_output.strip().replace("```json", "").replace("```", "")
    
    try:
        decision = json.loads(clean_json)
        
        if decision.get("is_math_course_related"):
            search_query = decision["search_query"]
            is_broad = decision.get("is_broad", False)
            
            # Step 2: Use Pinecone Metadata Filtering base on 'is_broad' factor
            if is_broad:
                print(f"  [System: Searching Pinecone for '{search_query}' (Broad/Summary)]")
                filter_dict = {"summary_type": {"$in": ["Class", "Professor"]}}
            else:
                print(f"  [System: Searching Pinecone for '{search_query}' (Specific)]")
                filter_dict = {"summary_type": {"$exists": False}}
                
            dynamic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5, "filter": filter_dict})
            
            rag_chain = (
                {
                    "context": lambda x: dynamic_retriever.invoke(x["question"]),
                    "question": lambda x: x["question"],
                    "history": lambda x: x["history"]
                } 
                | rag_prompt 
                | llm
            )
            
            return rag_chain.invoke({"question": search_query, "history": formatted_history}).content
        else:
            print("  [System: Routing to General Knowledge]")
            general_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Here is the chat history:\n{history}"),
                ("human", "{question}")
            ])
            return (general_prompt | llm).invoke({"question": user_input, "history": formatted_history}).content
            
    except json.JSONDecodeError:
        print("  [System: Routing to General (JSON Error)]")
        general_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Here is the chat history:\n{history}"),
            ("human", "{question}")
        ])
        return (general_prompt | llm).invoke({"question": user_input, "history": formatted_history}).content


# 6. Gradio Interface Preparation
def respond(message, history):
    parsed_history = []
    
    user_msg_cache = ""
    for item in history:
        if isinstance(item, (list, tuple)):
            parsed_history.append((item[0], item[1]))
        else:
            role = item.get("role") if isinstance(item, dict) else getattr(item, "role", "")
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", "")
            if role == "user":
                user_msg_cache = content
            elif role == "assistant":
                parsed_history.append((user_msg_cache, content))
                
    query_text = message.get("text", "") if isinstance(message, dict) else message
    
    return handle_query(query_text, parsed_history)

# Interface deployment configuration
with gr.Blocks() as demo:
    gr.Markdown("# UPD CRS Assistant")
    gr.Markdown("**Disclaimer:** Coverage is currently limited to UPD Math courses. Coverage will be widened as more courses are included.")
    gr.Markdown("<p style='font-size: 12px; color: gray;'>Double check the output, LLMs often make mistakes.</p>")
    
    gr.ChatInterface(fn=respond)

# Hugging Face deployment
if __name__ == "__main__":
    demo.launch()