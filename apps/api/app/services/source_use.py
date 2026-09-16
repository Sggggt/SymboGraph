"""Versioned content-role guard; leaves raw text, spans and graph facts intact."""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from app.retrieval_control_contracts import (
    EvidenceInterval, EvidenceScopeCoverage, EvidenceScopeExpression, EvidenceScopeFact, control_hash,
)


def _merge_scope_intervals(intervals):
    result = []
    for namespace, version, start, end in sorted(intervals):
        if result and result[-1][:2] == (namespace, version) and start <= result[-1][3]:
            previous = result[-1]
            result[-1] = (*previous[:2], previous[2], max(end, previous[3]))
        else:
            result.append((namespace, version, start, end))
    return tuple(result)


def _intersect_scope_intervals(left, right):
    if left is None:
        return right
    if right is None:
        return left
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        a, b = left[i], right[j]
        if a[:2] < b[:2]:
            i += 1
        elif a[:2] > b[:2]:
            j += 1
        else:
            start, end = max(a[2], b[2]), min(a[3], b[3])
            if start < end:
                result.append((*a[:2], start, end))
            if a[3] <= b[3]:
                i += 1
            if b[3] <= a[3]:
                j += 1
    return _merge_scope_intervals(result)


def _missing_scope_intervals(required, available):
    result, j = [], 0
    for namespace, version, start, end in required:
        key = namespace, version
        while j < len(available) and (available[j][:2] < key or
                available[j][:2] == key and available[j][3] <= start):
            j += 1
        cursor, k = start, j
        while k < len(available) and available[k][:2] == key and available[k][2] < end:
            cover = available[k]
            if cover[2] > cursor:
                result.append((*key, cursor, min(cover[2], end)))
            cursor = max(cursor, cover[3])
            if cursor >= end:
                break
            k += 1
        if cursor < end:
            result.append((*key, cursor, end))
    return tuple(result)


def evaluate_evidence_scope(*, knowledge_base_id: str, expression: EvidenceScopeExpression,
        scopes: tuple[EvidenceScopeFact, ...], packed: tuple[EvidenceInterval, ...], mode='overlap') -> EvidenceScopeCoverage:
    """Location coverage only. Resolvers and semantic sufficiency are separate."""
    if mode not in {'overlap', 'complete'} or len(scopes) > 64 or len(packed) > 4096:
        raise ValueError('evidence_scope_evaluation_capacity_or_mode_invalid')
    if any(item.knowledge_base_id != knowledge_base_id for item in (*scopes, *packed)):
        raise ValueError('evidence_scope_cross_knowledge_base')
    by_id = {scope.id: scope for scope in scopes}
    if len(by_id) != len(scopes):
        raise ValueError('evidence_scope_duplicate_identity')
    def ranges(items):
        return _merge_scope_intervals((item.knowledge_base_id, item.document_version_id, item.start, item.end) for item in items)
    def resolve(node):
        if node.op == 'scope':
            if node.scope_id not in by_id:
                raise ValueError('evidence_scope_reference_missing')
            fact = by_id[node.scope_id]
            lower = ranges(fact.intervals) if fact.resolution == 'verified' else ()
            upper = lower if fact.resolution == 'verified' and fact.extent_complete else None
            return lower, upper
        bounds = [resolve(child) for child in node.children]
        lower, upper = bounds[0]
        for next_lower, next_upper in bounds[1:]:
            if node.op == 'intersection':
                lower = _intersect_scope_intervals(lower, next_lower)
                upper = _intersect_scope_intervals(upper, next_upper)
            else:
                lower = _merge_scope_intervals((*lower, *next_lower))
                upper = None if upper is None or next_upper is None else _merge_scope_intervals((*upper, *next_upper))
        return lower, upper
    lower, upper = resolve(expression)
    available = ranges(packed)
    usable = _intersect_scope_intervals(lower, available)
    missing = _missing_scope_intervals(lower, available)
    if upper == ():
        state, reason = 'unsatisfied', 'empty_resolved_scope'
    elif mode == 'complete':
        if missing:
            state, reason = 'unsatisfied', 'required_range_missing'
        elif upper is None or not lower or _missing_scope_intervals(upper, available):
            state, reason = 'unknown', 'scope_not_fully_resolved'
        else:
            state, reason = 'satisfied', 'covered'
    elif usable:
        state, reason = 'satisfied', 'covered'
    elif _intersect_scope_intervals(upper, available) == ():
        state, reason = 'unsatisfied', 'no_scope_overlap'
    else:
        state, reason = 'unknown', 'scope_not_fully_resolved'
    def records(items):
        return tuple(EvidenceInterval(knowledge_base_id=kb, document_version_id=version, start=start, end=end)
                     for kb, version, start, end in items)
    identity = {'protocol_version':'evidence_scope_algebra_v1', 'knowledge_base_id':knowledge_base_id,
        'expression':expression.model_dump(mode='json'), 'scopes':[item.model_dump(mode='json')
            for item in sorted(scopes,key=lambda item:item.id)], 'packed':available, 'mode':mode}
    return EvidenceScopeCoverage(state=state,mode=mode,input_hash=control_hash(identity),
        usable_intervals=records(usable),missing_intervals=records(missing),scope_extent_known=upper is not None,reason=reason)


def combine_scope_states(states, *, op):
    """Combine independent obligations; union of ranges is not logical OR."""
    if not states or len(states) > 64 or op not in {'all','any'} or any(
            state not in {'satisfied','unsatisfied','unknown'} for state in states):
        raise ValueError('evidence_scope_obligation_invalid')
    decisive, uniform = ('unsatisfied','satisfied') if op == 'all' else ('satisfied','unsatisfied')
    if decisive in states:
        return decisive
    return uniform if all(state == uniform for state in states) else 'unknown'


PROTOCOL = "metadata_dominant_source_use_v2"
_YEAR = re.compile(r"\b(?:18|19|20)\d{2}\b")
_NUMBERED_REFERENCE = re.compile(r'^\d{1,4}[.)]\s+(?=[A-Z])')
_REFERENCE_PAGES = re.compile(r'\b\d{1,4}\s*:\s*[A-Z]?\d+\s*[–−-]\s*[A-Z]?\d+')
_LEADING_FRAGMENT = re.compile(r'^(?:of|and|which|whose|who|that|with|for)\b')
_VENUE = re.compile(r"\b(?:vol\.?|pp\.?|pages?|journal|proceedings|proc\.?|trans\.?|springer|press|conference|tech\.? rep\.?|ieee)\b", re.I)
_APA = re.compile(r"^[A-Z][\w'’.-]+,\s*[A-Z].{0,220}\((?:18|19|20)\d{2}[a-z]?\)")
_BIO = re.compile(r"\breceived\b.{0,120}\b(?:degree|Ph\.?D|B\.?S|M\.?S)|\bresearch interests\b|\bis (?:currently )?(?:an? )?(?:associate )?professor\b|\b(?:email addresses|manuscript received|manuscript TR\d|manuscripts received|authorized licensed use|digital object identifier)\b|作者简介|收稿日期", re.I)
_TECHNICAL_RESTART = re.compile(r"(?:The|This|Our)\s+(?:paper|algorithm|method|model|system)\b|本文|该算法|本方法", re.I)
_ASSERTION = re.compile(r"\b(?:is|are|has|have|can|will|must|show|shows|propose|proposes|define|defined|uses?|consists?|introduces?|evaluates?|provides?|requires?)\b|(?:采用|使用|基于|定义|包括|导致|满足|要求|说明|提出|能够|可以|表示)", re.I)
_METHOD = re.compile(r"\b(?:algorithm|method|technique|approach|procedure|using|uses|based on|via)\b|\b(?:avoid|overcome|solve|achieve)\b.{0,160}\bby\b|(?:算法|方法|技术|采用|使用|基于|通过)", re.I)
_COMPARISON = re.compile(r"\b(?:compared|than|lower|higher|simpler|less|more|better|worse|versus|advantage)\b|(?:相比|相对|优于|劣于|更低|更高|更少|更多|更简单|降低|提升)", re.I)
_CONDITION = re.compile(r"\b(?:if|when|must|require|requires|required|specified|outside|intervals?|non-overlapping|only)\b|(?:如果|当|必须|要求|满足|区间|范围|条件|约束)", re.I)
_METADATA_QUESTION = re.compile(r"参考文献|引用的.{0,12}(?:文献|论文|著作)|作者(?:是谁|有哪些|信息)|谁.{0,12}(?:提出|开发|发明)|\b(?:bibliograph\w*|references|cited (?:papers|works)|who (?:is|are))\b|\bwho\b.{0,35}\b(?:developed|invented|introduced)\b", re.I)


@dataclass(frozen=True)
class SourceUseAnalysis:
    prose: tuple[str, ...]
    total_characters: int
    metadata_characters: int

    @property
    def metadata_dominant(self):
        return self.metadata_characters * 5 >= self.total_characters * 3 and self.metadata_characters > 0


@dataclass(frozen=True)
class SourceUseDecision:
    allowed: bool
    metadata_dominant: bool
    reason: str
    form: str


def _metadata(text):
    if _BIO.search(text):
        return True
    if re.search(r"\(TR\d{2}-\d+\)", text) and text.startswith(('“', '"', '‘')):
        return True
    if re.match(r"^\[\d{1,3}\]", text) and (_YEAR.search(text) or _VENUE.search(text)):
        return True
    if (_NUMBERED_REFERENCE.match(text) and _YEAR.search(text)
            and (_VENUE.search(text) or _REFERENCE_PAGES.search(text) or 'doi.org/' in text)):
        return True
    apa = _APA.match(text)
    if apa:
        return not re.match(r"(?:showed|shows|demonstrated|proved|proposed|reported|argued|used)\b",
            text[apa.end():].lstrip('. '), re.I)
    # A continuation of a bibliographic entry has publication/page fields,
    # without a preceding substantive assertion.
    year = _YEAR.search(text)
    if year and _VENUE.search(text) and re.search(r"\b(?:vol|pp|pages)\.?\s*\d", text, re.I):
        return not _ASSERTION.search(text[:year.start()])
    return False


def analyze_source_use(text):
    parts = re.split(r"\n\s*\n|(?=\n(?:\[\d{1,3}\]\s|\d{1,4}[.)]\s+[A-Z]))", text)
    if len(parts) > 512:
        raise ValueError("source_use_segment_capacity_exceeded")
    prose, total, metadata = [], 0, 0
    for part in parts:
        normalized = " ".join(unicodedata.normalize("NFKC", part).split())
        if not normalized:
            continue
        # A PDF block may contain a biography followed by genuine prose.
        # Keep the latter; never discard an entire mixed block wholesale.
        bio = _BIO.search(normalized)
        restart = _TECHNICAL_RESTART.search(normalized, bio.end()) if bio else None
        cuts = {0, len(normalized)}
        if restart:
            cuts.add(restart.start())
        if bio:
            preceding_sentence = normalized.rfind('. ', 0, bio.start())
            if (preceding_sentence >= 0 and not _LEADING_FRAGMENT.match(normalized)
                    and (_ASSERTION.search(normalized[:preceding_sentence]) or _TECHNICAL_RESTART.match(normalized))):
                cuts.add(preceding_sentence + 2)
        ordered = sorted(cuts)
        pieces = tuple(normalized[left:right] for left, right in zip(ordered, ordered[1:]))
        for piece in pieces:
            total += len(piece)
            if _metadata(piece):
                metadata += len(piece)
            elif not piece.startswith('#'):
                prose.append(piece)
    return SourceUseAnalysis(tuple(prose), total, metadata)


def requested_form(task, facet):
    text = facet.text + (' ' + task.question if len(task.requirements) == 1 else '')
    if _METADATA_QUESTION.search(text):
        return 'metadata'
    if facet.role == 'comparison' or re.search(r"比较|相比|相对|优[势点]|\b(?:compare|comparison|versus|advantages?)\b", text, re.I):
        return 'comparison'
    if re.search(r"方法|算法|\b(?:method|algorithm|procedure|how to)\b", text, re.I) or facet.role == 'procedure':
        return 'method'
    if re.search(r"约束|条件|分布.{0,12}区间|\b(?:condition|constraint|distribution|interval)\b", text, re.I):
        return 'condition'
    if facet.role == 'quantity':
        return 'quantity'
    return 'assertion'


def source_use_decision(task, facet, analysis, *, roles=()):
    form = requested_form(task, facet)
    if not analysis.metadata_dominant:
        return SourceUseDecision(True, False, 'ordinary_source', form)
    if form == 'metadata':
        return SourceUseDecision(True, True, 'metadata_requested', form)
    prose = ' '.join(analysis.prose)
    structured = ('table' in roles and re.search(r'\d', prose)
        or 'formula' in roles and re.search(r'[=≤≥∑∫]|\\(?:frac|sum|prod)\b', prose)
        or 'code' in roles and re.search(r'\b(?:def|class|return|function)\b|[{};]', prose))
    if structured:
        return SourceUseDecision(True, True, 'structured_source', form)
    matcher = {'method': _METHOD, 'comparison': _COMPARISON, 'condition': _CONDITION,
        'assertion': _ASSERTION}.get(form)
    allowed = any((matcher.search(part) if matcher else re.search(r'\d', part) and _ASSERTION.search(part))
        for part in analysis.prose)
    return SourceUseDecision(bool(allowed), True,
        'substantive_form_present' if allowed else 'metadata_without_requested_form', form)
