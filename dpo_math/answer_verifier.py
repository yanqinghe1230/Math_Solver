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

import random
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


# ── Programmatic wrong answer generation (fallback) ──

def generate_wrong_cot(correct_cot: str) -> Optional[str]:
    """
    Programmatically generate a wrong variant of a correct CoT answer.

    Used as a last-resort fallback when the API cannot produce wrong answers
    (e.g., for very simple problems like '105 × 3 = ?').

    Applies one of several error strategies to make the reasoning internally
    consistent but ultimately arrive at an incorrect final answer.

    Returns the modified CoT text, or None if no transformation could be applied.
    """
    if not correct_cot or len(correct_cot) < 30:
        return None

    # Try strategies in random order until one succeeds
    strategies = [
        _error_arithmetic,
        _error_operator_swap,
        _error_number_shift,
        _error_missing_step,
        _error_wrong_operand,
    ]
    random.shuffle(strategies)

    for strategy_fn in strategies:
        result = strategy_fn(correct_cot)
        if result is not None and result != correct_cot:
            # Verify the result actually has a different final answer
            correct_gt = extract_ground_truth(correct_cot)
            wrong_gt = extract_ground_truth(result)
            if wrong_gt and is_answer_wrong(correct_gt, wrong_gt):
                return result

    return None


# ── Error strategies ──

def _error_arithmetic(cot: str) -> Optional[str]:
    """
    Find a calculation like 'X op Y = Z' and change Z to a wrong value
    while keeping the reasoning flow consistent.
    """
    # Match patterns like: "105 × 3 = 315" or "150 ÷ 5 = 30"
    pattern = re.compile(r'(\d+(?:\.\d+)?)\s*([×÷+\-])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)')
    matches = list(pattern.finditer(cot))
    if not matches:
        return None

    # Pick a match (prefer the last one — usually closest to final answer)
    match = random.choice(matches)
    a, op, b, correct_result = match.group(1), match.group(2), match.group(3), match.group(4)

    # Compute wrong result
    a_val = float(a)
    b_val = float(b)
    correct_val = float(correct_result)

    # Generate a plausible wrong answer
    wrong_val = _plausible_wrong(a_val, b_val, op, correct_val)
    if wrong_val is None:
        return None

    # Format the wrong result the same way as the original
    if '.' in correct_result:
        wrong_str = f"{wrong_val:.1f}" if wrong_val == int(wrong_val) else str(round(wrong_val, 2))
    elif correct_result.endswith('/') or '/' in str(correct_val):
        wrong_str = str(int(wrong_val))
    else:
        wrong_str = str(int(wrong_val))

    # Replace just this occurrence
    old = f"{a} {op} {b} = {correct_result}"
    new = f"{a} {op} {b} = {wrong_str}"
    cot = cot.replace(old, new, 1)

    # Also update any downstream calculation that uses the wrong result
    # Find the final answer and replace it
    correct_gt = extract_ground_truth(cot)  # This won't work because we just changed the calc but not the final answer yet

    # Actually, let's update the final answer section
    cot = _update_final_answer(cot, wrong_str)

    return cot


def _error_operator_swap(cot: str) -> Optional[str]:
    """
    Swap an arithmetic operator: × → +, ÷ → -, + → ×.
    Then recalculate the result accordingly.
    """
    # Find a calculation with an operator we can swap
    pattern = re.compile(r'(\d+(?:\.\d+)?)\s*([×÷+\-])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)')
    matches = list(pattern.finditer(cot))
    if not matches:
        return None

    match = random.choice(matches)
    a, op, b, correct_result = match.group(1), match.group(2), match.group(3), match.group(4)
    a_val, b_val = float(a), float(b)

    # Swap operators
    swaps = {'×': '+', '+': '×', '÷': '-', '-': '÷'}
    if op not in swaps:
        return None
    new_op = swaps[op]

    # Compute with the new operator
    try:
        if new_op == '+':
            wrong_val = a_val + b_val
        elif new_op == '×':
            wrong_val = a_val * b_val
        elif new_op == '-':
            wrong_val = a_val - b_val
        elif new_op == '÷':
            wrong_val = a_val / b_val if b_val != 0 else a_val
        else:
            return None
    except (ValueError, ZeroDivisionError):
        return None

    if wrong_val == float(correct_result):
        return None  # No actual change

    # Format
    wrong_str = str(int(wrong_val)) if wrong_val == int(wrong_val) else f"{wrong_val:.1f}"

    old = f"{a} {op} {b} = {correct_result}"
    new = f"{a} {new_op} {b} = {wrong_str}"
    cot = cot.replace(old, new, 1)
    cot = _update_final_answer(cot, wrong_str)

    return cot


def _error_number_shift(cot: str) -> Optional[str]:
    """
    Shift one of the input numbers slightly: 105 → 150, 150 → 105,
    120 → 102, then recalculate.
    """
    # Find a number that looks like it came from the problem
    pattern = re.compile(r'(\d+(?:\.\d+)?)\s*([×÷+\-])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)')
    matches = list(pattern.finditer(cot))
    if not matches:
        return None

    match = random.choice(matches)
    a, op, b, correct_result = match.group(1), match.group(2), match.group(3), match.group(4)
    a_val, b_val = float(a), float(b)

    # Decide which operand to shift (a or b)
    if random.random() < 0.5 and len(a) >= 2:
        # Digit swap: 105 → 150, 120 → 102
        new_a = _digit_swap(a)
        if new_a == a:
            return None
        a_val, new_str = float(new_a), new_a
        old_target, new_target = a, new_a
    elif len(b) >= 2:
        new_b = _digit_swap(b)
        if new_b == b:
            return None
        b_val, new_str = float(new_b), new_b
        old_target, new_target = b, new_b
    else:
        return None

    # Recalculate
    try:
        if op == '×':
            wrong_val = a_val * b_val
        elif op == '+':
            wrong_val = a_val + b_val
        elif op == '-':
            wrong_val = a_val - b_val
        elif op == '÷':
            wrong_val = a_val / b_val if b_val != 0 else a_val
        else:
            return None
    except (ValueError, ZeroDivisionError):
        return None

    wrong_str = str(int(wrong_val)) if wrong_val == int(wrong_val) else f"{wrong_val:.1f}"

    old = f"{old_target} {op} {match.group(3) if old_target == a else match.group(1)} = {correct_result}"
    new = f"{new_target} {op} {match.group(3) if old_target == a else match.group(1)} = {wrong_str}"
    cot = cot.replace(old, new, 1)
    cot = _update_final_answer(cot, wrong_str)

    return cot


def _error_missing_step(cot: str) -> Optional[str]:
    """
    Remove one reasoning step and adjust the final answer.
    Works by removing a numbered step line and replacing the answer
    with a value computed from the preceding step.
    """
    # Find numbered steps
    lines = cot.split('\n')
    step_pattern = re.compile(r'^\d+\.\s')
    step_indices = [i for i, line in enumerate(lines) if step_pattern.match(line.strip())]

    if len(step_indices) < 2:
        return None

    # Remove a middle step
    remove_idx = random.choice(step_indices[1:-1] if len(step_indices) > 2 else step_indices[1:])
    lines.pop(remove_idx)

    # Re-number remaining steps
    new_lines = []
    step_num = 1
    for line in lines:
        stripped = line.strip()
        if step_pattern.match(stripped):
            # Replace the step number
            new_line = re.sub(r'^\d+\.', f'{step_num}.', stripped, count=1)
            new_lines.append(new_line)
            step_num += 1
        else:
            new_lines.append(stripped)

    # Find the most recent calculation before the 最终答案 section
    # and use a different value as the final answer
    cot_modified = '\n'.join(new_lines)

    # Extract the last number in the reasoning
    reasoning_nums = re.findall(r'=\s*(\d+(?:\.\d+)?)', cot_modified)
    if reasoning_nums:
        last_calc = float(reasoning_nums[-1])
        # Offset it slightly
        wrong_val = last_calc + random.choice([-1, 1, -10, 10, -5, 5])
        if wrong_val <= 0:
            wrong_val = last_calc * random.choice([0.5, 0.8, 1.2, 1.5])
        wrong_str = str(int(wrong_val)) if wrong_val == int(wrong_val) else f"{wrong_val:.1f}"
        cot_modified = _update_final_answer(cot_modified, wrong_str)
        return cot_modified

    return None


def _error_wrong_operand(cot: str) -> Optional[str]:
    """
    Use a wrong operand value: pick a number from elsewhere in the problem
    instead of the correct one. This is the most subtle and natural-looking error.
    """
    # Find all calculations: "A op B = C"
    calc_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*([×÷+\-])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)')
    calc_matches = list(calc_pattern.finditer(cot))
    if not calc_matches:
        return None

    # Find all standalone numbers in the reasoning
    all_numbers = re.findall(r'(?<!\d)(\d+(?:\.\d+)?)(?!\d)', cot)
    if len(all_numbers) < 3:
        return None

    # Pick a random calculation
    match = random.choice(calc_matches)
    a, op, b, result = match.group(1), match.group(2), match.group(3), match.group(4)
    a_val, b_val = float(a), float(b)

    # Pick a replacement number different from a, b, and result
    candidates = [n for n in all_numbers if n != a and n != b and n != result]
    if not candidates:
        return None

    replacement = random.choice(candidates)
    repl_val = float(replacement)

    # Decide which operand to replace
    if random.random() < 0.5:
        # Replace a
        a_val, old_target, new_target = repl_val, a, replacement
    else:
        # Replace b
        b_val, old_target, new_target = repl_val, b, replacement

    # Recalculate with the wrong operand
    try:
        if op == '×':
            wrong_val = a_val * b_val
        elif op == '+':
            wrong_val = a_val + b_val
        elif op == '-':
            wrong_val = a_val - b_val
        elif op == '÷':
            wrong_val = a_val / b_val if b_val != 0 else a_val
        else:
            return None
    except (ValueError, ZeroDivisionError):
        return None

    if wrong_val == float(result):
        return None  # No change

    wrong_str = str(int(wrong_val)) if wrong_val == int(wrong_val) else f"{wrong_val:.1f}"

    # Replace the old operand with new operand in the matched span
    full_match = match.group(0)
    if old_target == a:
        new_calc = f"{new_target} {op} {b} = {wrong_str}"
    else:
        new_calc = f"{a} {op} {new_target} = {wrong_str}"

    cot = cot.replace(full_match, new_calc, 1)
    cot = _update_final_answer(cot, wrong_str)
    return cot


# ── Helpers ──

def _plausible_wrong(a: float, b: float, op: str, correct: float) -> Optional[float]:
    """
    Generate a plausible wrong answer by introducing a common arithmetic error.
    """
    candidates = []

    # Off-by-one in one digit of the result
    if correct != 0:
        candidates.append(correct + 1)
        candidates.append(correct - 1)

    # Digit transposition (e.g., 315 → 351, 135, 513)
    correct_int = int(correct)
    if correct == correct_int and correct_int >= 10:
        s = str(correct_int)
        if len(s) >= 3:
            # Swap adjacent digits
            i = random.randint(0, len(s) - 2)
            swapped = s[:i] + s[i+1] + s[i] + s[i+2:]
            if swapped != s:
                candidates.append(float(swapped))

    # Multiplication by wrong factor
    if op == '×':
        candidates.append(a * (b + random.choice([-2, -1, 1, 2])))
    elif op == '+':
        candidates.append(a + b + random.choice([-1, 1, -10, 10]))
    elif op == '-':
        candidates.append(abs(a - b + random.choice([-1, 1, -5, 5])))

    # Remove candidates that equal the correct answer
    candidates = [c for c in candidates if c != correct and c > 0]

    if not candidates:
        return correct + random.choice([1, 2, 5, -1, -2, -5])

    return random.choice(candidates)


def _digit_swap(s: str) -> str:
    """Swap two adjacent digits in a number string."""
    if len(s) < 2:
        return s
    i = random.randint(0, len(s) - 2)
    return s[:i] + s[i+1] + s[i] + s[i+2:]


def _update_final_answer(cot: str, new_answer: str) -> str:
    """
    Replace the final answer in a CoT text with a new value.
    Handles the standard ### 最终答案 format.
    """
    # Pattern: ### 最终答案\n{value}
    m = re.search(r'(#*\s*最终答案\s*\n\s*)(.+?)(\s*)$', cot, re.MULTILINE | re.DOTALL)
    if m:
        before = cot[:m.start(2)]
        after = cot[m.end(2):]
        return before + new_answer + after

    # Pattern: 最终答案是/為 {value}
    m = re.search(r'(最终答案[是为：:]\s*)(.+?)(\n|$)', cot)
    if m:
        before = cot[:m.start(2)]
        after = cot[m.end(2):]
        return before + new_answer + after

    # Fallback: append a wrong final answer line
    if '最终答案' not in cot and '推理过程' in cot:
        return cot.rstrip() + f"\n\n### 最终答案\n{new_answer}"

    return cot
