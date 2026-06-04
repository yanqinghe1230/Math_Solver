"""
Prompt templates for generating variant (potentially wrong) math answers.
Four strategies produce diverse error types for DPO training.
"""

# ── Strategy A: High temperature (natural stochastic errors) ──
SYSTEM_HIGH_TEMP = """你是一位小学数学老师。请逐步推理并给出最终答案。

输出格式要求：
### 推理过程
（写出你的逐步推理过程）

### 最终答案
（仅输出最终数字答案，不带单位）"""


# ── Strategy B: Common student mistakes (explicit error seeding) ──
SYSTEM_STUDENT_MISTAKE = """你是一名正在学习数学的小学生。请尝试解答下面的题目。

你可能会犯一些常见的学生错误，比如：
- 看错运算符号（把加看成减、把乘看成除）
- 忽略题目中的单位换算
- 计算过程中抄错数字
- 只做了一步推理就停止了
- 忘记题目问的是什么

请写出你的推理过程并给出最终答案。注意：你的推理过程要看起来像是认真思考过的，但里面包含一个不易察觉的错误。

输出格式要求：
### 推理过程
（写出你的逐步推理过程）

### 最终答案
（仅输出最终数字答案，不带单位）"""


# ── Strategy C: Rushed test-taker (careless errors) ──
SYSTEM_RUSHED = """你正在参加一场数学竞赛，时间非常紧迫，只剩下最后1分钟。请快速解答下面的题目，不需要检查答案是否正确。

输出格式要求：
### 推理过程
（快速写出推理过程）

### 最终答案
（仅输出最终数字答案，不带单位）"""


# ── Strategy D: Alternative solution path ──
SYSTEM_ALTERNATIVE = """请用与常规解法不同的另一种解题思路重新解答这道题。尽量从不同的角度思考。

输出格式要求：
### 推理过程
（写出你的逐步推理过程）

### 最终答案
（仅输出最终数字答案，不带单位）"""


# ── Strategy metadata ──
STRATEGIES = {
    "high_temp": {
        "system": SYSTEM_HIGH_TEMP,
        "temperature": 0.9,
        "description": "正常prompt + 高温，自然产生随机错误",
    },
    "student_mistake": {
        "system": SYSTEM_STUDENT_MISTAKE,
        "temperature": 0.8,
        "description": "扮演犯错小学生，故意犯常见错误",
    },
    "rushed": {
        "system": SYSTEM_RUSHED,
        "temperature": 0.9,
        "description": "扮演赶时间竞赛，粗心错误",
    },
    "alternative": {
        "system": SYSTEM_ALTERNATIVE,
        "temperature": 0.7,
        "description": "不同解题思路，可能正确也可能错",
    },
}


def build_messages(question: str, strategy: str) -> list[dict]:
    """
    Build chat messages for the given strategy.

    Args:
        question: The math problem text.
        strategy: One of 'high_temp', 'student_mistake', 'rushed', 'alternative'.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from: {list(STRATEGIES.keys())}")

    system_prompt = STRATEGIES[strategy]["system"]

    # Add the output format reminder to all strategies for consistent parsing
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    return messages


def get_strategy_temperature(strategy: str) -> float:
    """Get the recommended temperature for a given strategy."""
    return STRATEGIES[strategy]["temperature"]
