import os
from dotenv import load_dotenv
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

DB_FAISS_PATH = "vectorstore/db_faiss"

# تحميل قاعدة البيانات
embedding = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
db = FAISS.load_local(DB_FAISS_PATH, embedding, allow_dangerous_deserialization=True)

# إعداد LLM من Groq (أقوى نموذج مجاني منطقي)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.1-70b-versatile"

llm = ChatGroq(
    model=GROQ_MODEL_NAME,
    temperature=0.1,
    max_tokens=512,
    api_key=GROQ_API_KEY
)

# بناء سلسلة RAG
prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combine_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs={"k": 5}), combine_chain)

# اختبار سؤال
user_query = input("💬 اكتب سؤالك هنا: ")
response = rag_chain.invoke({"input": user_query})

print("\n🤖 الإجابة:")
print(response["answer"])

print("\n📚 المستندات المصدرية:")
for doc in response["context"]:
    print(f"- {doc.metadata} -> {doc.page_content[:200]}...")
