from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

model = init_chat_model("google_genai:gemini-3.5-flash-lite")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="rag_collection",
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 3,
        "fetch_k" : 10,
        "lambda_mult" : 0.5
    }
    
)



#prompt template 
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("Rag system created ")

print("press 0 to exit ")

while True:
    query = input("You : ")
    if query == "0":
        break 
    
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context" :context,
        "question": query
    })
    
    response = model.invoke(final_prompt)

    print("\nAI:")

    for block in response.content:
        if isinstance(block, dict) and block.get("type") == "text":
            print(block["text"])
        elif isinstance(block, str):
            print(block)
    

