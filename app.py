import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from langchain_community.document_loaders import GoogleDriveLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

app = Flask(__name__)

# ดึงค่าจาก Environment Variables (ตั้งค่าใน Render ภายหลัง)
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# ชื่อไฟล์ credential ที่จะอัปโหลดขึ้น Render
CREDENTIALS_PATH = "/etc/secrets/credentials.json" 

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# --- ส่วนของ RAG (Google Drive + Gemini) ---
# ประกาศตัวแปร Global ไว้เก็บระบบ RAG
qa_chain = None

def init_rag_system():
    global qa_chain
    print("กำลังเริ่มระบบ RAG... (อาจใช้เวลาสักครู่)")
    
    # ตรวจสอบว่ามีไฟล์ Credential ไหม (สำหรับรัน Local ให้เปลี่ยน path ได้)
    if not os.path.exists(CREDENTIALS_PATH):
        # กรณีรันในเครื่องตัวเองอาจจะอยู่ที่โฟลเดอร์เดียวกัน
        local_path = "credentials.json"
        if os.path.exists(local_path):
            auth_path = local_path
        else:
            print("ไม่พบไฟล์ credentials.json! บอทจะตอบได้แค่เรื่องทั่วไป")
            return
    else:
        auth_path = CREDENTIALS_PATH

    # 1. โหลดไฟล์จาก Drive
    try:
        loader = GoogleDriveLoader(
            folder_id=os.getenv('GOOGLE_DRIVE_FOLDER_ID'),
            credentials_path=auth_path,
            token_path="token.json", 
            recursive=True
        )
        docs = loader.load()
        
        # 2. แบ่งไฟล์และทำ Index
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # 3. สร้าง Chain สำหรับตอบคำถาม
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY),
            retriever=vectorstore.as_retriever(),
            return_source_documents=False
        )
        print("ระบบ RAG พร้อมใช้งานแล้ว!")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการโหลด Google Drive: {e}")

# รันฟังก์ชันโหลดข้อมูลทันทีเมื่อ Server เริ่มทำงาน
# หมายเหตุ: บน Render Free Tier อาจจะช้าหน่อยตอนเปิดครั้งแรก
init_rag_system()

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@app.route("/")
def home():
    return "Hello, Line Bot is running!"

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_text = event.message.text
    
    if qa_chain:
        try:
            # ให้ Gemini ค้นคำตอบจากเอกสาร
            reply_text = qa_chain.run(user_text)
        except Exception as e:
            reply_text = "ขออภัย เกิดข้อผิดพลาดในการประมวลผลเอกสาร"
            print(e)
    else:
        reply_text = "ระบบเอกสารยังไม่พร้อมใช้งาน (ตรวจสอบ Credential หรือ Folder ID)"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

if __name__ == "__main__":
    app.run()

