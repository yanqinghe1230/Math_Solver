"""
Answer extraction, normalization, and comparison utilities.

Handles the diverse answer formats found in Chinese elementary math problems:
- Integers: 315
- Decimals: 7.5
- Fractions: 4/5, 1/5
- Mixed fractions (Chinese notation): 2_1/5 (= 2 + 1/5)
- Percentages: 94.4%
- Boolean: 够/不够, 能/不能
- Multiple answers: 9/10,1/10; 20;28
- Numbers with units: 315千克, 7元
"""

import re
from typing import Optional, Union


# ── Common Chinese math units (stripped during normalization) ──
_UNITS = [
    "千克", "克", "吨", "公斤", "斤", "两",
    "米", "厘米", "分米", "毫米", "千米", "公里",
    "元", "角", "分",
    "个", "件", "只", "本", "支", "张", "块", "条", "辆", "艘",
    "棵", "株", "朵", "粒", "片", "双", "对", "套", "台", "部",
    "分钟", "小时", "秒", "天", "日", "周", "年", "次",
    "平方米", "平方分米", "平方厘米", "平方千米", "公顷",
    "立方米", "立方分米", "立方厘米",
    "升", "毫升",
    "千米/时", "米/秒", "米/分",
    "页", "题", "步", "层", "圈",
    "倍", "人",
    "万",
]


def extract_final_answer(text: str) -> Optional[str]:
    """
    Extract the final answer from a model-generated CoT text.

    Tries multiple patterns in order of reliability:
    1. ### 最终答案\\n{value} — the standard training format
    2. 答案是/答案为/答：{value} — common Chinese answer markers
    3. 最终答案是/最终答案为 — variations
    4. Last non-empty line — fallback

    Returns None if no answer can be extracted.
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # Pattern 1: ### 最终答案 followed by the answer on next line(s)
    m = re.search(r'#*\s*最终答案\s*\n\s*(.+?)\s*$', text, re.MULTILINE | re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        # Take the first meaningful line
        lines = [l.strip() for l in candidate.split('\n') if l.strip()]
        if lines:
            return _clean_answer_line(lines[0])

    # Pattern 2: 最终答案是/最終答案為
    m = re.search(r'最终答案[是为：:]\s*(.+?)(?:\n|$)', text)
    if m:
        return _clean_answer_line(m.group(1).strip())

    # Pattern 3: 答案是/答案为/答：/答:
    m = re.search(r'(?:答案(?:是|为)|答)[：:]\s*(.+?)(?:\n|$)', text)
    if m:
        return _clean_answer_line(m.group(1).strip())

    # Pattern 4: **答：** ... (bold format)
    m = re.search(r'\*\*答[：:]\*\*\s*(.+?)(?:\n|$)', text)
    if m:
        return _clean_answer_line(m.group(1).strip())

    # Pattern 5: 综上所述/因此/所以 ... 答案为 ...
    m = re.search(r'(?:综上所述|因此|所以|故).*?(?:答案|结果)(?:是|为|：|:)\s*(.+?)(?:\n|$)', text)
    if m:
        return _clean_answer_line(m.group(1).strip())

    # Fallback: last non-empty line (excluding markdown separators)
    lines = [l.strip() for l in text.split('\n') if l.strip() and not l.strip().startswith('---')]
    if lines:
        last = lines[-1]
        # Skip if it's just a markdown header
        if not last.startswith('#') and len(last) < 100:
            return _clean_answer_line(last)

    return None


def _clean_answer_line(line: str) -> str:
    """Remove common formatting noise from an extracted answer line."""
    # Remove markdown bold markers
    line = re.sub(r'\*\*', '', line)
    # Remove trailing punctuation
    line = line.rstrip('。，,;；.！!？?）)')
    # Remove leading bullet markers
    line = re.sub(r'^[-*•]\s*', '', line)
    return line.strip()


def normalize_answer(answer: str) -> Optional[Union[float, str]]:
    """
    Normalize an answer string for comparison.

    Returns:
        float — if the answer can be parsed as a number
        str   — if the answer is non-numeric (e.g., 够/不够, 能/不能)
        None  — if unparseable
    """
    if not answer or not answer.strip():
        return None

    text = answer.strip()

    # ── Boolean / categorical answers ──
    text_lower = text.lower().rstrip('.。！!')
    if text_lower in ('够', '能', '是', '对', '可以', 'yes', 'true'):
        return text_lower
    if text_lower in ('不够', '不能', '否', '不对', '不可以', 'no', 'false'):
        return text_lower

    # ── Remove units ──
    # Sort by length (longest first) to avoid partial matches
    sorted_units = sorted(_UNITS, key=len, reverse=True)
    for unit in sorted_units:
        text = text.replace(unit, '')

    text = text.strip()

    # ── Remove percentage sign ──
    is_percent = '%' in text or '％' in text
    text = text.replace('%', '').replace('％', '').strip()

    # ── Handle mixed fractions with underscore: 2_1/5 = 2 + 1/5 ──
    mixed_frac_match = re.match(r'^(-?\d+)_(\d+)\s*/\s*(\d+)$', text)
    if mixed_frac_match:
        whole = int(mixed_frac_match.group(1))
        num = int(mixed_frac_match.group(2))
        den = int(mixed_frac_match.group(3))
        if den != 0:
            sign = -1 if whole < 0 else 1
            return sign * (abs(whole) + num / den)

    # ── Handle simple fractions: 3/4 ──
    frac_match = re.match(r'^(-?\d+)\s*/\s*(\d+)$', text)
    if frac_match:
        num = int(frac_match.group(1))
        den = int(frac_match.group(2))
        if den != 0:
            return num / den

    # ── Handle pure number ──
    try:
        return float(text)
    except ValueError:
        pass

    # ── Chinese number conversion for small numbers (一 ~ 十, 百) ──
    cn_result = _parse_chinese_number(text)
    if cn_result is not None:
        return float(cn_result)

    # ── If text still contains non-numeric content after unit removal, return as string ──
    # This handles multi-part answers like "9/10,1/10" or "20;28"
    if re.search(r'[,;，；\s]', text):
        return text.lower()

    # Try cleaning more aggressively
    cleaned = re.sub(r'[^\d.\-/]', '', text)
    if cleaned:
        try:
            return float(eval(cleaned))  # eval handles expressions like "2-1/5"
        except (ValueError, SyntaxError, ZeroDivisionError):
            pass

    return text.lower() if text.lower() else None


def _parse_chinese_number(text: str) -> Optional[int]:
    """Parse Chinese number words like 一百二十 into integers."""
    CN_NUMS = {
        '零': 0, '〇': 0,
        '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
    }
    # Simple cases: just a single digit
    if text in CN_NUMS:
        return CN_NUMS[text]

    # Check if composed entirely of Chinese number characters
    if not all(c in '零〇一二两三四五六七八九十百千' for c in text):
        return None

    # Handle "X十Y" pattern (e.g., 三十五 = 35)
    if '十' in text:
        parts = text.split('十')
        if len(parts) == 2:
            tens = CN_NUMS.get(parts[0], 1) if parts[0] else 1
            ones = CN_NUMS.get(parts[1], 0) if parts[1] else 0
            return tens * 10 + ones
        elif len(parts) == 1:
            return CN_NUMS.get(parts[0], 1) * 10 if parts[0] else 10

    # Handle "X百Y十Z" pattern
    result = 0
    if '百' in text:
        bp = text.split('百')
        result += (CN_NUMS.get(bp[0], 1) if bp[0] else 1) * 100
        text = bp[1] if len(bp) > 1 else ''
    if '十' in text:
        sp = text.split('十')
        tens = CN_NUMS.get(sp[0], 1) if sp[0] else 1
        result += tens * 10
        text = sp[1] if len(sp) > 1 else ''
    if text and text in CN_NUMS:
        result += CN_NUMS[text]

    return result if result > 0 else None


def is_answer_wrong(ground_truth: str, generated: str, tolerance: float = 1e-9) -> bool:
    """
    Compare a generated answer against the ground truth.

    Returns:
        True  — the generated answer DIFFERS from ground truth (usable as rejected)
        False — the generated answer MATCHES ground truth (not useful)
        None  — cannot determine (one or both answers unparseable)
    """
    gt = normalize_answer(ground_truth)
    gen = normalize_answer(generated)

    # Both unparseable — cannot determine
    if gt is None and gen is None:
        return None

    # One unparseable — likely different
    if gt is None or gen is None:
        return True

    # Both numeric — compare with tolerance
    if isinstance(gt, (int, float)) and isinstance(gen, (int, float)):
        return abs(gt - gen) > tolerance

    # Both strings — exact match after normalization
    if isinstance(gt, str) and isinstance(gen, str):
        return gt != gen

    # Mixed types (one numeric, one string) — likely different
    return True


# ── Convenience function ──

def extract_ground_truth(answer_text: str) -> str:
    """
    Extract the ground truth answer from a train_cot.json answer field.

    This is the same extraction logic but optimized for the known format
    of the training data (### 最终答案\\n{value}).
    """
    m = re.search(r'最终答案\s*\n\s*(.+?)$', answer_text, re.MULTILINE)
    if m:
        return _clean_answer_line(m.group(1).strip())

    # Fallback for non-standard records
    return extract_final_answer(answer_text) or answer_text.strip()
