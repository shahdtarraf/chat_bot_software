import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

DB_FAISS_PATH = "vectorstore/db_faiss"
os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)

data_folder = "data"  # ضع ملفات البيانات هنا
docs = []

# تحميل ملفات النصوص
for filename in os.listdir(data_folder):
    if filename.endswith((".txt", ".pdf")):
        file_path = os.path.join(data_folder, filename)
        print(f"📘 جاري تحميل الملف: {filename}")
        try:
            if filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
                docs.extend(loader.load())
            elif filename.endswith(".pdf"):
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())
        except Exception as e:
            print(f"⚠️ خطأ في تحميل {filename}: {e}")

if docs:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    texts = text_splitter.split_documents(docs)
    
    print("⚙️ جاري تحميل نموذج التضمين...")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
    except Exception as e:
        print(f"⚠️ خطأ في تحميل النموذج: {e}")
        print("استخدام نموذج بديل...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
    
    print("⚙️ جاري إنشاء قاعدة بيانات FAISS...")
    db = FAISS.from_documents(texts, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print("✅ تم إنشاء قاعدة بيانات FAISS بنجاح!")
else:
    print("❌ لم يتم العثور على ملفات بيانات في مجلد data")
