"""
Prompt template for generating variant (wrong) math answers via Zhipu API.

Single strategy: simulate a weaker student who preserves most reasoning steps
but introduces one key error, resulting in a wrong final answer.
"""

# ── System prompt: simulate a weaker student ──
SYSTEM_WEAK_STUDENT = """下面是一道题和正确推理。

请模拟一个能力较弱的学生。
要求：
1. 保留大部分推理步骤
2. 在一个关键步骤产生错误
3. 最终答案错误
4. 不要检查和修正错误

输出格式要求：
### 推理过程
（写出你的逐步推理过程）

### 最终答案
（仅输出最终数字答案，不带单位）"""


def build_messages(question: str, correct_cot: str) -> list[dict]:
    """
    Build chat messages for the weak-student strategy.

    Includes both the question and the correct CoT as context,
    so the model can reference the reasoning steps while introducing an error.

    Args:
        question: The math problem text.
        correct_cot: The correct chain-of-thought answer from train_cot.json.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    user_content = f"题目：{question}\n\n正确推理：\n{correct_cot}"

    messages = [
        {"role": "system", "content": SYSTEM_WEAK_STUDENT},
        {"role": "user", "content": user_content},
    ]
    return messages


def get_strategy_temperature() -> float:
    """Get the recommended temperature for the weak-student strategy."""
    return 0.8
