import re
from sympy import sympify
from sympy.parsing.latex import parse_latex


def extract_boxed(text):
    if not text:
        return None
    matches = re.findall(r"\\boxed\s*{([^}]*)}", text)
    return matches[-1].strip() if matches else None

def extract_all_boxed(text):
    if not text:
        return []
    return [m.strip() for m in re.findall(r"\\boxed\s*{([^}]*)}", text)]

def normalize_latex(ans):
    if ans is None:
        return None
    ans = ans.strip()
    ans = ans.replace(r"\left", "").replace(r"\right", "")
    ans = ans.replace(r"\,", "") 
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    ans = ans.replace("^\\circ", "")
    ans = ans.replace("°", "")
    ans = " ".join(ans.split())
    return ans


def normalize(ans):
    if ans is None:
        return None
    ans = ans.replace("$", "")
    ans = ans.replace(r"\left", "").replace(r"\right", "")
    ans = ans.replace(r"\,", "")
    ans = ans.replace(r"\pi", "pi")
    ans = re.sub(r"\\frac{([^}]*)}{([^}]*)}", r"(\1)/(\2)", ans)
    ans = ans.replace("np.", "")
    ans = re.sub(r"\s+", "", ans)
    return ans


def to_int(ans):
    if ans is None:
        return None
    ans = ans.strip()
    ans = re.sub(r"[^\d\-]", "", ans)
    try:
        return int(ans)
    except Exception:
        return None

def split_gold_answers(gold_list):
    results = []
    for ans in gold_list:
        if ans is None:
            continue
        ans = ans.replace("$", "")
        for p in ans.split(","):
            p = normalize(p)
            if p:
                results.append(p)
    return results


def split_pred(pred):
    if pred is None:
        return []
    return [normalize(p) for p in pred.split(",") if p.strip()]


def try_numeric(x):
    try:
        return float(x)
    except Exception:
        return None


def math_equal_symbolic(pred, gold):
    if pred == gold:
        return True
    try:
        p = parse_latex(pred)
    except Exception:
        try:
            p = sympify(pred)
        except Exception:
            p = None
    try:
        g = parse_latex(gold)
    except Exception:
        try:
            g = sympify(gold)
        except Exception:
            g = None
    if p is None or g is None:
        return False
    try:
        return p.equals(g)
    except Exception:
        return False


def math_equal(pred, gold):
    if pred == gold:
        return True
    p_num = try_numeric(pred)
    g_num = try_numeric(gold)
    if p_num is not None and g_num is not None:
        return abs(p_num - g_num) / max(1, abs(g_num)) < 1e-6
    try:
        return sympify(pred) == sympify(gold)
    except Exception:
        return False
