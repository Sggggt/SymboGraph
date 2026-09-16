"""Keep explicit response instructions out of the evidence requirement set."""
import re

from app.retrieval_control_contracts import ResponseConstraint

PROTOCOL = 'user_response_constraints_v1'
_QUESTION = re.compile(r'如何|怎么|什么|是否|为什么|何时|多少|哪[些个种]|\b(?:how|what|why|whether|which|does|did|is there)\b', re.I)
_SPECULATION = r'(?:不要|不得|不能|不应|不)(?:自行)?(?:推测|猜测|编造|臆测)|\b(?:do not|don.t|must not|never)\s+(?:guess|speculate|invent|fabricate)\b'
_INSUFFICIENCY = r'(?:没有|无(?:足够)?证据|未找到|证据不足|资料不足)(?:就|时|则|的情况下).{0,24}(?:说明|告知|说|否定)|\b(?:if|when)\b.{0,50}\b(?:no evidence|not found|insufficient)\b.{0,35}\b(?:say|state|explain|report)\b'
_FORMAT = r'(?:请)?(?:用|以|按).{0,12}(?:表格|列表|JSON|Markdown|格式).{0,12}(?:输出|回答|列出|展示|组织)|\b(?:answer|respond|output|present)\b.{0,25}\b(?:table|json|markdown|bullet)\b'
_LANGUAGE = r'(?:请)?(?:用|以)(?:中文|英文|英语|汉语)(?:回答|说明|输出)|\b(?:answer|respond) in (?:english|chinese)\b'
_BREVITY = r'(?:请)?(?:简洁|简短|简明)(?:地)?(?:回答|说明|表述)|\b(?:keep (?:the answer|it) (?:short|brief)|answer briefly)\b'
_KINDS = [('no_speculation', re.compile(_SPECULATION, re.I)),
          ('insufficiency_notice', re.compile(_INSUFFICIENCY, re.I)),
          ('output_format', re.compile(_FORMAT, re.I)),
          ('language', re.compile(_LANGUAGE, re.I)), ('brevity', re.compile(_BREVITY, re.I))]
_PURE_START = re.compile(r'^(?:请|不要|不得|不能|不应|无(?:足够)?证据|证据不足|资料不足|没有|未找到|若无|若没有|回答(?:时|应|需)|输出(?:格式|应|需)|\b(?:do not|don.t|never|if|when|answer|respond|output|keep|no evidence)\b)', re.I)
_DIRECTIVE_START = re.compile(r'^(?:请)?(?:不要|不得|不能|不应|无(?:足够)?证据|证据不足|资料不足|没有|未找到|若无|若没有|用|以|按|简洁|简短|简明)|\b(?:please\s+)?(?:do not|don.t|never|if|when|answer|respond|output|present|keep)\b', re.I)


def response_constraints(question):
    result = []
    # These are input instruction spans, never answer-sentence entailment tests.
    candidates = []
    for sentence in re.finditer(r'[^。！？!?；;\n]+', question):
        candidates.append((sentence.start(), sentence.end()))
        candidates.extend((sentence.start()+part.start(), sentence.start()+part.end())
            for part in re.finditer(r'[^，,]+', sentence.group()))
        candidates.extend((sentence.start()+part.end(), sentence.end()) for part in
            re.finditer(r'\.\s+(?=(?:Do not|Don.t|Never|Please|If|When|Answer|Respond|Keep)\b)', sentence.group(), re.I))
    for start, end in sorted(set(candidates)):
        raw = question[start:end]
        left = start + len(raw) - len(raw.lstrip())
        right = end - len(raw) + len(raw.rstrip())
        text = question[left:right]
        if text and _DIRECTIVE_START.match(text) and not _QUESTION.search(text):
            for kind, pattern in _KINDS:
                if pattern.search(text):
                    result.append(ResponseConstraint(kind=kind, text=text, char_span=(left, right)))
    result = [item for item in result if not any(other.kind == item.kind
        and item.char_span[0] <= other.char_span[0] < other.char_span[1] <= item.char_span[1]
        and other.char_span != item.char_span for other in result)]
    if len(result) > 12:
        raise ValueError('task_response_constraint_capacity_exceeded')
    return tuple(result)


def split_response_requirements(question, requirements):
    constraints = response_constraints(question)
    kinds = {item.kind for item in constraints}
    kept, moved = [], []
    for requirement in requirements:
        text = requirement.facet.strip()
        matches = {kind for kind, pattern in _KINDS if pattern.search(text)}
        purely_response = (bool(_PURE_START.search(text)) and not _QUESTION.search(text)
            and bool(matches) and matches.issubset(kinds)
            and not requirement.protected_literals and not requirement.source_roles and not requirement.source_scope
            and re.search(r'\d|规定|条款|政策|条件|阈值|原因|理由|查明|提取|核实|解释|分析|计算|\b(?:threshold|reason|policy|explain|calculate)\b', text, re.I) is None)
        if purely_response:
            moved.append(text)
        else:
            kept.append(requirement)
    if not kept:
        raise ValueError('task_has_no_evidence_requirement')
    return tuple(kept), constraints, {'protocol_version': PROTOCOL,
        'response_constraint_count': len(constraints), 'response_only_facets_removed': len(moved),
        'evidence_requirement_count': len(kept), 'original_question_preserved': True}
