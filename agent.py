import os              
import sqlite3          # Kết nối và thao tác database SQLite (không cần cài server riêng)
from pathlib import Path  

from dotenv import load_dotenv  # Đọc file .env và nạp API key, config vào os.environ
import certifi                  # Cung cấp chứng chỉ SSL/TLS chuẩn để verify HTTPS an toàn

# Tải các biến môi trường từ file .env (ví dụ: GEMINI_API_KEY, TAVILY_API_KEY,...)
load_dotenv()

# Trỏ chứng chỉ SSL đến file chứng chỉ của certifi để tránh lỗi SSL khi gọi API bên ngoài
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_google_genai import ChatGoogleGenerativeAI
# Định dạng tin nhắn hệ thống (System Prompt) cho mô hình
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
# ToolNode: Node chạy các công cụ; tools_condition: Điều kiện rẽ nhánh kiểm tra xem LLM có gọi tool không
from langgraph.prebuilt import ToolNode, tools_condition
# SqliteSaver: Lưu trạng thái và lịch sử hội thoại vào SQLite để duy trì ngữ cảnh nhiều lượt chat
from langgraph.checkpoint.sqlite import SqliteSaver
# Danh sách các công cụ hỗ trợ của agent (web search, calculator, memory, RAG...) được định nghĩa từ file tools.py
from tools import tools


Path("data").mkdir(exist_ok=True)


# Update default and allowed models to use Gemini 2.5
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

ALLOWED_MODELS = {
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite", # Phiên bản gọn nhẹ, tốc độ cao
    "gemini-1.5-flash",      # Giữ lại để dự phòng / tương thích ngược
    "gemini-1.5-pro"
}


# Lời nhắc hệ thống: Quy định vai trò, nhiệm vụ và các quy tắc ứng xử/gọi tool cho AI Agent
SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant named Tit similar to ChatGPT.

You can:
1. Answer normal questions.
2. Use tools when needed.
3. Search uploaded documents using the RAG tool.
4. Search the web for latest/current information using Tavily Search.
5. Remember important user information using the memory tool.
6. Recall memory when useful.
7. Use calculator for math.

Rules:
- If the user asks about latest news, current events, recent updates, today's information, current prices, current people, current versions, new releases, or anything time-sensitive, use Tavily Search.
- If the user asks about an uploaded document, use search_uploaded_documents.
- If the user asks you to remember something, use remember_this.
- If the user asks about previous preferences or saved facts, use recall_memory.
- Use calculator for math questions.
- When using web search, summarize clearly and mention that the answer is based on web search results.
- Be clear, helpful, and concise.
"""



def normalize_model_name(model_name: str | None) -> str:
    """
    HÀM 1: CHUẨN HÓA VÀ KIỂM TRA TÊN MODEL (Validation & Fallback)
    - Tác dụng:
      + Nhận tên model mà người dùng / frontend truyền lên.
      + Loại bỏ khoảng trắng thừa.
      + Kiểm tra xem model có nằm trong danh sách cho phép (ALLOWED_MODELS) hay không.
      + Nếu không truyền hoặc tên model không hợp lệ, tự động fallback về DEFAULT_MODEL để tránh gây lỗi hệ thống.
    """

    if not model_name:
        return DEFAULT_MODEL

    model_name = model_name.strip()

    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name




def build_agent(model_name: str):
    """
    HÀM 2: KHỞI TẠO VÀ BIÊN DỊCH AGENT CHO MỘT MODEL CỤ THỂ (LangGraph Agent Builder)
    - Tác dụng:
      + Tạo kết nối với mô hình Gemini đã chọn thông qua ChatGoogleGenerativeAI.
      + Gắn (bind) danh sách các công cụ (tools) vào model để mô hình biết nó có thể gọi tool nào.
      + Xây dựng đồ thị trạng thái hội thoại (StateGraph):
        * chatbot_node: Node tiếp nhận tin nhắn, ghép System Prompt và gọi LLM trả lời hoặc yêu cầu gọi tool.
        * tool_node: Node tiếp nhận lệnh và thực thi các tool khi chatbot_node yêu cầu.
        * Luồng chạy: START -> chatbot -> kiểm tra tools_condition:
          - Nếu cần gọi tool -> chuyển sang node 'tools' -> chạy xong quay về 'chatbot' để tổng hợp câu trả lời.
          - Nếu không cần gọi tool -> kết thúc lượt xử lý.
      + Thiết lập SqliteSaver để lưu trạng thái phiên chat (checkpoint/memory) vào SQLite.
      + Trả về workflow đã được compile sẵn sàng hoạt động.
    """

    # Đảm bảo tên model luôn hợp lệ
    selected_model = normalize_model_name(model_name)

    # Initialize ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=0.3,    # Giảm tính sáng tạo ngẫu nhiên, ưu tiên câu trả lời chính xác, bám sát sự thật
        streaming=True       # Hỗ trợ stream phản hồi theo thời gian thực về giao diện
    )

    # Khai báo các công cụ (tools) cho mô hình để LLM có thể chủ động kích hoạt Function Calling
    llm_with_tools = llm.bind_tools(tools)

    def chatbot_node(state: MessagesState):
        """
        Node xử lý hội thoại của Chatbot:
        - Ghép SYSTEM_PROMPT vào đầu danh sách tin nhắn để luôn duy trì vai trò và chỉ dẫn hệ thống.
        - Gọi LLM kèm theo các công cụ đã bind để sinh ra câu trả lời hoặc lệnh gọi tool.
        - Trả về tin nhắn mới để cập nhật vào trạng thái (state).
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response]
        }

    # Node phụ trách thực thi các tool khi mô hình yêu cầu
    tool_node = ToolNode(tools)

    # Khởi tạo đồ thị trạng thái quản lý chuỗi tin nhắn MessagesState
    workflow = StateGraph(MessagesState)

    # Đăng ký các node vào đồ thị
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tool_node)

    # Thiết lập luồng di chuyển giữa các node:
    # 1. Bắt đầu từ START -> đi vào node 'chatbot'
    workflow.add_edge(START, "chatbot")
    # 2. Điều kiện rẽ nhánh: nếu LLM yêu cầu gọi tool -> chuyển sang node 'tools', nếu không -> kết thúc lượt
    workflow.add_conditional_edges("chatbot", tools_condition)
    # 3. Sau khi node 'tools' chạy xong -> trả kết quả về lại node 'chatbot' để LLM đọc và trả lời người dùng
    workflow.add_edge("tools", "chatbot")

    # Mở kết nối đến database SQLite để lưu lịch sử và checkpoints của các phiên chat
    conn = sqlite3.connect(
        "data/langgraph_checkpoints.sqlite",
        check_same_thread=False  # Cho phép nhiều thread dùng chung kết nối (phù hợp với môi trường web server)
    )

    # Tạo checkpointer dùng SQLite để lưu và khôi phục trạng thái cuộc trò chuyện theo session
    checkpointer = SqliteSaver(conn)

    # Biên dịch đồ thị luồng làm việc thành runnable agent có gắn bộ nhớ checkpoint
    return workflow.compile(checkpointer=checkpointer)


# Bộ nhớ đệm (Cache) lưu trữ các agent đã được khởi tạo để tái sử dụng, tránh tạo lại nhiều lần gây tốn tài nguyên
_AGENT_CACHE = {}


def get_agent(model_name: str | None = None):
    """
    HÀM 3: LẤY HOẶC TẠO MỚI AGENT THEO CƠ CHẾ CACHING (Agent Factory / Singleton Pattern)
    - Tác dụng:
      + Kiểm tra xem Agent ứng với `selected_model` đã từng được khởi tạo trong `_AGENT_CACHE` chưa.
      + Nếu ĐÃ CÓ: Lấy ngay từ cache ra dùng (không mất thời gian kết nối và biên dịch lại).
      + Nếu CHƯA CÓ: Gọi hàm `build_agent(selected_model)` để tạo mới, lưu vào cache rồi trả về.
      + Đảm bảo mỗi loại model chỉ cần compile đồ thị 1 lần duy nhất trong suốt vòng đời ứng dụng.
    """

    selected_model = normalize_model_name(model_name)

    # Nếu model này chưa từng được tạo agent thì khởi tạo và đưa vào cache
    if selected_model not in _AGENT_CACHE:
        _AGENT_CACHE[selected_model] = build_agent(selected_model)

    # Trả về instance agent từ bộ nhớ cache
    return _AGENT_CACHE[selected_model]