"""
Module AI Chatbot - Trợ lý tài chính cá nhân (RAG + Gemini API)
Nâng cấp từ mô hình Rule-based NLU sang Retrieval-Augmented Generation (RAG)
kết hợp Google Gemini API qua LangChain (langchain-google-genai).
"""

import os
import logging
from dotenv import load_dotenv

from datetime import datetime
from collections import defaultdict
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from data import (
    SAMPLE_TRANSACTIONS,
    SAMPLE_BUDGETS,
    SAMPLE_GOALS,
    CATEGORIES_BY_ID,
    DEFAULT_CATEGORIES,
)
from analytics import (
    forecast_spending,
    detect_recurring_bills,
    survival_days,
)

logger = logging.getLogger(__name__)

# ===================== 1. CẤU HÌNH BIẾN MÔI TRƯỜNG & GEMINI API =====================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# Cấu hình LLM tối ưu Tốc độ: gemini-3.5-flash-lite (sub-second latency)
DEFAULT_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.1-flash-lite"]
DEFAULT_TIMEOUT = 20
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_OUTPUT_TOKENS = 250

class AIQuotaOrBusyError(Exception):
    """Lỗi khi AI bị quá tải 429 (Resource Exhausted) hoặc 503 (Unavailable)."""
    pass

class AITimeoutError(Exception):
    """Lỗi khi gọi AI vượt quá thời gian timeout (20s)."""
    pass

try:
    llm = ChatGoogleGenerativeAI(
        model=DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        google_api_key=GOOGLE_API_KEY,
        timeout=DEFAULT_TIMEOUT,
    )
except Exception as e:
    logger.warning("Lỗi khởi tạo llm ban đầu: %s", e)
    llm = None


# ===================== 2. TIỆN ÍCH TRÍCH XUẤT NGỮ CẢNH RÚT GỌN (LIGHTWEIGHT RAG) =====================
def _format_vnd(amount) -> str:
    """Format tiền VND: 1500000 -> '1.500.000 đ'."""
    try:
        return f"{int(amount):,}".replace(",", ".") + " đ"
    except (ValueError, TypeError):
        return f"{amount} đ"


def build_lightweight_context(user_query: str) -> str:
    """
    Tối ưu RAG Context Filtering & Chunking:
    - Phân loại và chỉ lọc dữ liệu cần thiết dựa trên nội dung câu hỏi.
    - Tuyệt đối không truyền toàn bộ lịch sử giao dịch thô nhằm giảm tối đa Input Tokens,
      giúp đạt tốc độ phản hồi nhanh nhất (sub-second / low-latency).
    """
    q = (user_query or "").lower().strip()
    now = datetime.now()
    cur_month = now.strftime("%Y-%m")

    # 1. Tính toán 3 chỉ số tài chính cốt lõi tháng hiện tại
    month_tx = [t for t in SAMPLE_TRANSACTIONS if t.get("date", "").startswith(cur_month)]
    total_income = sum(t["amount"] for t in month_tx if t.get("type") == "income")
    total_expense = sum(t["amount"] for t in month_tx if t.get("type") == "expense")
    balance = total_income - total_expense

    core_summary = (
        f"Tháng {now.strftime('%m/%Y')}: Thu {_format_vnd(total_income)} | "
        f"Chi {_format_vnd(total_expense)} | Dư {_format_vnd(balance)}"
    )

    # 2. Câu hỏi liên quan tới "ngân sách" / "hạn mức" -> Chỉ truyền danh sách ngân sách khả dụng
    if any(k in q for k in ["ngân sách", "hạn mức", "định mức", "budget"]):
        budget_lines = []
        for b in SAMPLE_BUDGETS:
            cat = CATEGORIES_BY_ID.get(b.get("category_id"), {})
            c_name = cat.get("name", b.get("category_id", "Khác"))
            budget_lines.append(f"- {c_name}: {_format_vnd(b.get('limit', 0))}/tháng")
        return f"[HẠN MỨC NGÂN SÁCH]:\n" + "\n".join(budget_lines)

    # 3. Câu hỏi liên quan tới "mục tiêu" / "tiết kiệm" -> Chỉ truyền danh sách mục tiêu tiết kiệm
    if any(k in q for k in ["mục tiêu", "tiết kiệm", "goal"]):
        goal_lines = []
        for g in SAMPLE_GOALS:
            pct = round(g.get("saved", 0) / g.get("target", 1) * 100, 1)
            goal_lines.append(
                f"- {g.get('name')}: {_format_vnd(g.get('saved', 0))}/{_format_vnd(g.get('target', 0))} ({pct}%)"
            )
        return f"[MỤC TIÊU TIẾT KIỆM]:\n" + "\n".join(goal_lines)

    # 4. Câu hỏi liên quan tới "giao dịch" / "chi tiêu" -> Chỉ truyền 5 giao dịch gần nhất
    if any(k in q for k in ["giao dịch", "chi tiêu", "khoản chi", "mua", "ăn", "tiền"]):
        recent_tx = sorted(SAMPLE_TRANSACTIONS, key=lambda x: x.get("date", ""), reverse=True)[:5]
        tx_lines = [
            f"- [{t.get('date')}] {t.get('description')}: {_format_vnd(t.get('amount', 0))}"
            for t in recent_tx
        ]
        return f"{core_summary}\n[5 GIAO DỊCH GẦN NHẤT]:\n" + "\n".join(tx_lines)

    # 5. Mặc định: Chỉ truyền đúng 3 chỉ số cốt lõi [Tổng thu tháng], [Tổng chi tháng], [Số dư hiện tại]
    return f"[TỔNG QUAN TÀI CHÍNH CỐT LÕI]:\n{core_summary}"


def build_rag_context() -> dict:
    """Hàm tương thích ngược với code cũ nếu có gọi."""
    return {"context": build_lightweight_context("")}


# ===================== 3. SYSTEM PROMPT TỐI ƯU TỐC ĐỘ =====================
SYSTEM_PROMPT = """Bạn là trợ lý tài chính. Trả lời trực tiếp, ngắn gọn trong tối đa 2-3 câu. Đi thẳng vào số liệu chính, không chào hỏi dài dòng, không giải thích rườm rà.
Dữ liệu tài chính:
{context}"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{user_message}"),
])


def _is_rate_limit_or_busy(err: Exception) -> bool:
    """Kiểm tra xem lỗi có phải do 429 (Resource Exhausted) hoặc 503 (Unavailable) không."""
    err_str = str(err).lower()
    return any(k in err_str for k in [
        "429", "503", "resource_exhausted", "resource exhausted",
        "unavailable", "quota", "too many requests"
    ])


def _is_timeout(err: Exception) -> bool:
    """Kiểm tra lỗi timeout."""
    err_str = str(err).lower()
    return any(k in err_str for k in ["timeout", "timed out", "deadline", "timedout"])


# ===================== 4. HÀM THỰC THI CHAIN & FALLBACK =====================
def _execute_chain(input_data: dict) -> str:
    """Thực thi LangChain Chain với mô hình Gemini Flash Lite tốc độ cao (timeout 20s) và xử lý fallback."""
    global llm

    # Đảm bảo có LLM với Flash Lite và max_output_tokens 250
    if llm is None:
        llm = ChatGoogleGenerativeAI(
            model=DEFAULT_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            google_api_key=GOOGLE_API_KEY,
            timeout=DEFAULT_TIMEOUT,
        )

    # Chain chuẩn: prompt_template | llm | StrOutputParser
    chain = prompt_template | llm | StrOutputParser()

    try:
        return chain.invoke(input_data)
    except Exception as e:
        err_msg = str(e)
        logger.warning("Lỗi invoke với %s: %s", DEFAULT_MODEL, err_msg)

        # Nếu lỗi 429 (Resource Exhausted) hoặc 503 (Unavailable), lập tức raise để xử lý
        if _is_rate_limit_or_busy(e):
            raise AIQuotaOrBusyError(err_msg)

        # Nếu lỗi Timeout
        if _is_timeout(e):
            raise AITimeoutError(err_msg)

        # Fallback các model tương đương nếu model chính gặp lỗi
        if any(k in err_msg.lower() for k in ["not found", "404", "unsupported", "unavailable", "503", "no longer available"]):
            for fb_model in FALLBACK_MODELS:
                try:
                    logger.info("Thử fallback sang model: %s (timeout: %ss)", fb_model, DEFAULT_TIMEOUT)
                    fb_llm = ChatGoogleGenerativeAI(
                        model=fb_model,
                        temperature=DEFAULT_TEMPERATURE,
                        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                        google_api_key=GOOGLE_API_KEY,
                        timeout=DEFAULT_TIMEOUT,
                    )
                    fb_chain = prompt_template | fb_llm | StrOutputParser()
                    return fb_chain.invoke(input_data)
                except Exception as fb_err:
                    if _is_rate_limit_or_busy(fb_err):
                        raise AIQuotaOrBusyError(str(fb_err))
                    if _is_timeout(fb_err):
                        raise AITimeoutError(str(fb_err))
                    logger.warning("Fallback model %s thất bại: %s", fb_model, fb_err)
                    continue

        raise e


# ===================== 5. HYBRID INTENT ENGINE (LOCAL ANALYTICS MAPPING) =====================

def _handle_spending_analysis() -> str:
    """Xử lý phân tích chi tiêu tháng này trực tiếp từ dữ liệu data.py."""
    now = datetime.now()
    cur_month = now.strftime("%Y-%m")
    month_name = now.strftime("%m/%Y")

    month_tx = [t for t in SAMPLE_TRANSACTIONS if t.get("date", "").startswith(cur_month)]
    total_income = sum(t["amount"] for t in month_tx if t.get("type") == "income")
    total_expense = sum(t["amount"] for t in month_tx if t.get("type") == "expense")
    balance = total_income - total_expense
    savings_rate = round(balance / total_income * 100, 1) if total_income > 0 else 0

    cat_expenses = defaultdict(int)
    for t in month_tx:
        if t.get("type") == "expense":
            cat_expenses[t.get("category_id")] += t.get("amount", 0)

    sorted_cats = sorted(cat_expenses.items(), key=lambda x: x[1], reverse=True)

    lines = [
        f"📊 **Báo cáo Phân tích Chi tiêu Tháng {month_name}:**\n",
        f"- **Tổng thu nhập:** {_format_vnd(total_income)}",
        f"- **Tổng chi tiêu:** {_format_vnd(total_expense)}",
        f"- **Số dư dòng tiền:** {_format_vnd(balance)} (Tỷ lệ tích lũy: `{savings_rate}%`)\n",
        f"📌 **Chi tiêu theo từng danh mục:**",
    ]

    if sorted_cats:
        for cat_id, amount in sorted_cats:
            cat = CATEGORIES_BY_ID.get(cat_id, {})
            c_name = cat.get("name", cat_id)
            c_icon = cat.get("icon", "💸")
            pct = round(amount / total_expense * 100, 1) if total_expense > 0 else 0
            lines.append(f"  • {c_icon} **{c_name}:** {_format_vnd(amount)} (`{pct}%`)")
    else:
        lines.append("  • Chưa có giao dịch chi tiêu nào trong tháng này.")

    exp_tx = [t for t in month_tx if t.get("type") == "expense"]
    if exp_tx:
        top_tx = sorted(exp_tx, key=lambda x: x.get("amount", 0), reverse=True)[:3]
        lines.append("\n🔍 **Top các khoản chi lớn nhất:**")
        for t in top_tx:
            lines.append(f"  • [{t.get('date')}] **{t.get('description')}:** {_format_vnd(t.get('amount', 0))}")

    return "\n".join(lines)


def _handle_recurring_bills() -> str:
    """Xử lý thông tin hóa đơn định kỳ sắp đến hạn từ detect_recurring_bills()."""
    bills = detect_recurring_bills()
    if not bills:
        return (
            "📋 **Hóa đơn định kỳ & Khoản chi lặp lại:**\n\n"
            "Hiện tại hệ thống chưa phát hiện khoản hóa đơn định kỳ nào cần thanh toán."
        )

    lines = [
        "📋 **Danh sách Hóa đơn định kỳ & Khoản chi sắp đến hạn:**\n",
    ]

    monthly_est = sum(b.get("avg_amount", 0) for b in bills)
    lines.append(f"💡 **Tổng chi phí định kỳ ước tính:** {_format_vnd(monthly_est)} / tháng\n")

    for b in bills:
        label = b.get("label", "Hóa đơn")
        icon = b.get("category_icon", "🧾")
        amount = _format_vnd(b.get("avg_amount", 0))
        cycle = b.get("cycle_text", "tháng")
        next_date = b.get("next_date_str") or "Chưa xác định"
        days_until = b.get("days_until")

        status = ""
        if b.get("is_imminent"):
            status = " 🚨 **Gấp: Trong 3 ngày!**"
        elif b.get("is_upcoming"):
            status = " ⚠️ **Sắp đến hạn**"

        due_info = f"còn {days_until} ngày" if days_until is not None and days_until >= 0 else "đã quá hạn"
        lines.append(
            f"- {icon} **{label}:** {amount} ({cycle}) | Hạn kế tiếp: **{next_date}** ({due_info}){status}"
        )

    return "\n".join(lines)


def _handle_forecast_spending() -> str:
    """Xử lý dự báo chi tiêu cuối tháng từ forecast_spending()."""
    res = forecast_spending()
    if not res.get("has_data"):
        return f"🔮 **Dự báo chi tiêu cuối tháng:**\n\n{res.get('message', 'Chưa đủ dữ liệu để dự báo.')}"

    month = res.get("month", "")
    spent = _format_vnd(res.get("spent_so_far", 0))
    avg_per_day = _format_vnd(res.get("avg_per_day", 0))
    speed_7day = _format_vnd(res.get("speed_7day", 0))
    forecast = _format_vnd(res.get("forecast_total", 0))
    budget = _format_vnd(res.get("budget_total", 0))
    diff = res.get("diff_vs_budget")

    lines = [
        f"🔮 **Dự báo Chi tiêu Cuối tháng ({month}):**\n",
        f"- **Đã chi tiêu đến nay:** {spent} (sau {res.get('days_elapsed', 0)} ngày)",
        f"- **Tốc độ chi bình quân:** {avg_per_day}/ngày (7 ngày gần nhất: {speed_7day}/ngày)",
        f"- **Dự báo tổng chi cuối tháng:** **{forecast}**",
        f"- **Tổng ngân sách đề ra:** {budget}",
    ]

    if diff is not None:
        if diff > 0:
            lines.append(f"- ⚠️ **Dự kiến vượt ngân sách:** +{_format_vnd(diff)}")
            if res.get("overshoot_date"):
                lines.append(f"- 🚨 **Thời điểm chạm trần ngân sách:** {res.get('overshoot_date')}")
        else:
            lines.append(f"- ✅ **Dự kiến tiết kiệm dưới hạn mức:** {_format_vnd(abs(diff))}")

    return "\n".join(lines)


def _handle_survival_fund() -> str:
    """Xử lý phân tích quỹ dự phòng và số ngày sinh tồn từ survival_days()."""
    res = survival_days()
    if not res.get("has_data"):
        return f"🛡️ **Quỹ dự phòng & Khả năng sinh tồn:**\n\n{res.get('message', 'Chưa đủ dữ liệu để tính toán.')}"

    balance = _format_vnd(res.get("balance_available", 0))
    essential_day = _format_vnd(res.get("essential_per_day", 0))
    essential_month = _format_vnd(res.get("essential_total_30d", 0))
    days = res.get("survival_days", 0)
    level = res.get("level", "safe")

    status_map = {
        "safe": "🟢 **An toàn** - Quỹ dự phòng đủ cho sinh hoạt cơ bản trên 60 ngày.",
        "watch": "🟡 **Cần theo dõi** - Duy trì được 30 - 60 ngày chi phí thiết yếu.",
        "warning": "🟠 **Cảnh báo** - Dưới 30 ngày chi tiêu thiết yếu, nên hạn chế mua sắm phát sinh.",
        "danger": "🔴 **Nguy hiểm** - Số dư cạn kiệt hoặc âm, cần bổ sung quỹ khẩn cấp ngay!",
    }
    status_text = status_map.get(level, "Bình thường")
    months_equiv = round(days / 30, 1)

    lines = [
        "🛡️ **Đánh giá Quỹ Dự phòng & Số Ngày Sinh tồn:**\n",
        f"- **Số dư dòng tiền khả dụng:** {balance}",
        f"- **Chi phí thiết yếu trung bình:** {essential_day}/ngày (~{essential_month}/tháng)",
        f"- **Thời gian duy trì chi tiêu:** **`{days} ngày`** (~ `{months_equiv} tháng`)",
        f"- **Trạng thái an toàn:** {status_text}",
    ]
    return "\n".join(lines)


def _handle_savings_goals() -> str:
    """Xử lý tra cứu tiến độ mục tiêu tiết kiệm từ SAMPLE_GOALS."""
    if not SAMPLE_GOALS:
        return "🎯 **Mục tiêu tiết kiệm:** Hiện tại bạn chưa thiết lập mục tiêu tiết kiệm nào."

    lines = [
        "🎯 **Tiến độ các Mục tiêu Tiết kiệm của bạn:**\n",
    ]

    total_saved = 0
    total_target = 0

    for g in SAMPLE_GOALS:
        name = g.get("name", "Mục tiêu")
        target = g.get("target", 0)
        saved = g.get("saved", 0)
        total_saved += saved
        total_target += target

        pct = round(saved / target * 100, 1) if target > 0 else 0
        remain = max(0, target - saved)

        bar_len = 10
        filled = min(int(pct / 10), bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        lines.append(
            f"- 📌 **{name}:**\n"
            f"  • Đã tích lũy: {_format_vnd(saved)} / {_format_vnd(target)} (**`{pct}%`**)\n"
            f"  • Còn thiếu: {_format_vnd(remain)}\n"
            f"  • Tiến độ: `[{bar}] {pct}%`"
        )

    total_pct = round(total_saved / total_target * 100, 1) if total_target > 0 else 0
    lines.append(
        f"\n💰 **Tổng cộng:** Đã tiết kiệm {_format_vnd(total_saved)} / {_format_vnd(total_target)} "
        f"(`{total_pct}%` trên toàn bộ mục tiêu)."
    )
    return "\n".join(lines)


def resolve_local_intent(query: str):
    """
    So khớp câu hỏi với các Intent mẫu phân tích tài chính nội bộ.
    Trả về tuple (intent_name, response_text) nếu khớp; ngược lại trả về None.
    """
    if not query:
        return None

    # Chuẩn hóa chuỗi: loại bỏ emoji ở đầu chuỗi và khoảng trắng thừa
    clean = re.sub(r"^[^\w\s]+", "", query).strip().lower()

    # 1. Phân tích chi tiêu tháng này
    spending_patterns = [
        "phân tích chi tiêu tháng này",
        "phân tích chi tiêu",
        "thống kê chi tiêu tháng",
        "chi tiêu tháng này",
        "tổng chi tiêu tháng",
        "báo cáo chi tiêu tháng",
        "tháng này chi nhiều nhất vào đâu",
        "chi nhiều nhất vào đâu",
    ]
    if any(p in clean for p in spending_patterns):
        return ("spending_analysis", _handle_spending_analysis())

    # 2. Hóa đơn định kỳ sắp đến hạn
    recurring_patterns = [
        "hóa đơn định kỳ sắp đến hạn",
        "hóa đơn định kỳ",
        "hóa đơn sắp đến hạn",
        "hóa đơn sắp tới",
        "khoản chi định kỳ",
        "hóa đơn cần đóng",
    ]
    if any(p in clean for p in recurring_patterns):
        return ("recurring_bills", _handle_recurring_bills())

    # 3. Dự báo chi tiêu cuối tháng
    forecast_patterns = [
        "dự báo chi tiêu cuối tháng",
        "dự báo chi tiêu",
        "chi tiêu cuối tháng",
        "dự đoán chi tiêu",
        "ước tính chi tiêu cuối tháng",
    ]
    if any(p in clean for p in forecast_patterns):
        return ("forecast_spending", _handle_forecast_spending())

    # 4. Quỹ dự phòng của tôi
    survival_patterns = [
        "quỹ dự phòng của tôi",
        "quỹ dự phòng",
        "số ngày sinh tồn",
        "quỹ khẩn cấp",
        "sinh tồn bao nhiêu ngày",
    ]
    if any(p in clean for p in survival_patterns):
        return ("survival_fund", _handle_survival_fund())

    # 5. Tiến độ mục tiêu tiết kiệm
    goals_patterns = [
        "tiến độ mục tiêu tiết kiệm",
        "mục tiêu tiết kiệm",
        "tiến độ tiết kiệm",
        "các mục tiêu tiết kiệm",
        "mục tiêu mua laptop",
        "tiết kiệm laptop",
    ]
    if any(p in clean for p in goals_patterns):
        return ("savings_goals", _handle_savings_goals())

    return None


# ===================== 6. HÀM XỬ LÝ CHÍNH KẾT NỐI VỚI APP.PY =====================
def chat(message: str, history=None) -> dict:
    """
    Hàm xử lý chat trả về JSON dict:
    - Nếu khớp local intent: status = "success"
    - Nếu gặp 429 (Resource Exhausted) hoặc 503 (Unavailable):
      trả về {"status": "error", "reply": "⚠️ Hệ thống AI đang bận hoặc chạm giới hạn lượt hỏi. Vui lòng thử lại sau 1 phút!"}
    - Nếu timeout (> 8s): status = "error", reply = "⚠️ Yêu cầu xử lý quá 8 giây. Hệ thống AI đang phản hồi chậm, vui lòng thử lại sau!"
    - Nếu thành công: status = "success", reply = ...
    """
    user_msg_clean = (message or "").strip()
    if not user_msg_clean:
        return {
            "status": "success",
            "reply": "Xin chào! Bạn có thể đặt câu hỏi về các khoản chi tiêu, thu nhập, ngân sách hoặc mục tiêu tiết kiệm của mình.",
            "intent": "greeting",
            "confidence": 1.0,
        }

    # 1. Kiểm tra Intent mẫu local (Mô hình Hybrid)
    local_match = resolve_local_intent(user_msg_clean)
    if local_match:
        intent_name, local_reply = local_match
        logger.info("Hybrid Match: Khớp local intent '%s', trả lời trực tiếp từ dữ liệu nội bộ.", intent_name)
        return {
            "status": "success",
            "reply": local_reply,
            "intent": intent_name,
            "confidence": 1.0,
        }

    # 2. Nếu không khớp Intent mẫu -> Kích hoạt RAG Chain gọi Gemini API
    if not GOOGLE_API_KEY:
        return {
            "status": "error",
            "reply": "⚠️ **Lỗi cấu hình**: Chưa tìm thấy `GOOGLE_API_KEY` trong file `.env`.\n- Vui lòng thêm `GOOGLE_API_KEY=your_api_key` vào file `.env` để kích hoạt AI Chatbot.",
            "intent": "rag_gemini",
            "confidence": 0.0,
        }

    # 2.1 Trích xuất ngữ cảnh RAG rút gọn (Context Filtering & Chunking)
    try:
        context_str = build_lightweight_context(user_msg_clean)

        input_data = {
            "context": context_str,
            "user_message": user_msg_clean,
        }

        # 2.2 Thực thi Chain Flash Lite với input tokens tối thiểu
        reply = _execute_chain(input_data)
        return {
            "status": "success",
            "reply": reply.strip(),
            "intent": "rag_gemini",
            "confidence": 1.0,
        }
    except AIQuotaOrBusyError:
        logger.warning("Gemini API gặp lỗi 429 (Resource Exhausted) hoặc 503 (Unavailable).")
        return {
            "status": "error",
            "reply": "⚠️ Hệ thống AI đang bận hoặc chạm giới hạn lượt hỏi. Vui lòng thử lại sau 1 phút!",
        }
    except AITimeoutError:
        logger.warning("Gemini API vượt quá thời gian phản hồi (timeout 20s).")
        return {
            "status": "error",
            "reply": "⚠️ Yêu cầu xử lý quá thời gian quy định (20s). Vui lòng thử lại câu hỏi ngắn hơn!",
        }
    except Exception as e:
        logger.exception("Lỗi chi tiết khi gọi Gemini API trong chat: %s", e)
        if _is_rate_limit_or_busy(e):
            return {
                "status": "error",
                "reply": "⚠️ Hệ thống AI đang bận hoặc chạm giới hạn lượt hỏi. Vui lòng thử lại sau 1 phút!",
            }
        if _is_timeout(e):
            return {
                "status": "error",
                "reply": "⚠️ Yêu cầu xử lý quá thời gian quy định (20s). Vui lòng thử lại câu hỏi ngắn hơn!",
            }
        return {
            "status": "error",
            "reply": f"❌ **Không thể kết nối đến AI lúc này**: {str(e)}\n\n- Vui lòng kiểm tra lại kết nối mạng hoặc thử lại sau.",
        }


def get_bot_response(user_id: str, user_message: str) -> str:
    """
    Hàm tiện ích lấy nội dung reply dạng văn bản từ hàm chat().
    """
    res = chat(message=user_message)
    return res.get("reply", "")


# ===================== SANITY CHECK =====================
if __name__ == "__main__":
    test_questions = [
        "Xin chào, bạn có thể giúp gì cho tôi?",
        "Tháng này tôi đã chi tiêu những khoản nào?",
        "Mục tiêu tiết kiệm mua Laptop của tôi đã đạt bao nhiêu phần trăm?",
    ]
    print("=" * 60)
    print("Kiểm tra trích xuất RAG Context:")
    ctx = build_rag_context()
    print("Transactions sample:", ctx["transactions"][:150], "...")
    print("Budgets sample:", ctx["budgets"][:150], "...")
    print("Goals sample:", ctx["goals"][:150], "...")
    print("=" * 60)

    for q in test_questions:
        print(f"\n[Người dùng]: {q}")
        res = get_bot_response(user_id="demo", user_message=q)
        print(f"[AI Chatbot]:\n{res}\n")
