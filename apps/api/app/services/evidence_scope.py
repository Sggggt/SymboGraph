"""Resolve declarative source locations against one admitted text/structure view.

The resolver grants location facts only. It does not rank graph paths, prove
semantic answers, or treat missing text as proof about the original document.
"""
from __future__ import annotations

from collections import defaultdict
from bisect import bisect_left
import math
import re
import unicodedata

from sqlalchemy import select

from app.models import ChunkStructureNode
from app.retrieval_control_contracts import (
    EvidenceInterval, EvidenceScopeExpression, EvidenceScopeFact, FacetScopeInput,
    FacetScopeStatus, SourceScopeBinding, control_hash,
)
from app.services.source_use import (
    _intersect_scope_intervals, _merge_scope_intervals, _missing_scope_intervals, combine_scope_states, evaluate_evidence_scope,
)
from app.services.storage import raise_if_source_io_cancelled
from app.services.structure_roles import structure_roles,normalized_role_title,SUMMARY_TITLES,CONTENTS_TITLES

PROTOCOL = 'structure_scope_resolver_v1'
MAX_NODES = 16384
MAX_TEXT_WINDOW = 4096
MAX_CACHED_CHARACTERS = 8 * 1024 * 1024
_NODE_KINDS = {'document': 'document', 'section': 'section', 'paragraph': 'text',
    'table': 'table', 'formula': 'formula', 'code_block': 'code', 'caption': 'caption', 'figure': 'figure'}
_KIND_WORDS = {
    'document': r'document|paper|report|文档|论文|报告',
    'section': r'section|chapter|章节|小节|章节内容',
    'text': r'text|body|prose|正文|文本',
    'table': r'tables?|表|表格', 'formula': r'equations?|formulae?|formulas?|公式|方程',
    'code': r'code|source code|algorithm|listing|代码|源码|源代码|算法', 'figure': r'figures?|images?|图|图片|图像',
    'caption': r'captions?|图注|表注',
}


def _normal(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split()).strip('# ')


def _title_matches(reference, title):
    wanted, title = _normal(reference), _normal(title)
    if not wanted:
        return False
    left = r'(?<![a-z0-9_])' if wanted[0].isascii() and wanted[0].isalnum() else ''
    right = r'(?![a-z0-9_])' if wanted[-1].isascii() and wanted[-1].isalnum() else ''
    return re.search(left + re.escape(wanted) + right,title) is not None


def _document_alias_matches(reference, title):
    """Match stable display/file title aliases without consulting content."""

    def tokens(value):
        result = []
        for token in re.findall(
            r"[a-z0-9]+|[\u3400-\u9fff]+",
            unicodedata.normalize("NFKC", str(value or "")).casefold(),
        ):
            if token.isdigit() and len(token) == 4:
                continue
            if token.endswith("ies") and len(token) > 4:
                token = token[:-3] + "y"
            elif token.endswith("s") and len(token) > 3:
                token = token[:-1]
            result.append(token)
        return set(result)

    wanted, candidate = tokens(reference), tokens(title)
    shared = wanted & candidate
    if len(wanted) == 1:
        token = next(iter(wanted))
        return token in candidate and token not in {
            "document", "file", "paper", "report", "manual", "survey",
            "文档", "文件", "论文", "报告", "手册", "巡天",
        }
    return (
        len(wanted) >= 2
        and len(candidate) >= 2
        and len(shared) >= 2
        and len(shared) / min(len(wanted), len(candidate)) >= 0.6
    )


def _label(text):
    # A representation grammar shared by every object kind; never answer terms.
    match = re.match(r'^(?:[^\W\d_]+[\s.]*|第)?(\d+(?:\.\d+)*|[IVXLCDM]+)(?=$|[\s:：.、)\]章节目])',
                     unicodedata.normalize('NFKC', text).strip('# \t\n'), re.I)
    return match.group(1).casefold() if match else None


def validate_scope_declarations(obligation):
    stack = [obligation]
    while stack:
        node = stack.pop()
        stack.extend(node.children)
        if hasattr(node, 'scope') and node.scope is not None:
            stack.append(node.scope)
        selector = getattr(node, 'selector', None)
        if selector is None:
            continue
        if selector.match == 'label' and _label(selector.reference) is None:
            raise ValueError('source_scope_label_declaration_invalid')
        if selector.match == 'kind' and re.fullmatch(_KIND_WORDS[selector.kind], _normal(selector.reference), re.I) is None:
            raise ValueError('source_scope_kind_must_be_explicit')


def coverage_requests(obligation):
    if obligation.op == 'coverage':
        return (obligation,)
    return tuple(item for child in obligation.children for item in coverage_requests(child))


class StructureScopeIndex:
    def __init__(self, *, corpus, nodes, task_hash=None):
        if len(nodes) > MAX_NODES or len(corpus.sources) > 32768:
            raise ValueError('source_scope_structure_capacity_exceeded')
        self.corpus = corpus
        self.task_hash = task_hash
        self.nodes = tuple(nodes)
        self.by_version = defaultdict(list)
        for source in corpus.sources:
            self.by_version[source.document_version_id].append(source)
        self.ends = {}
        for version, sources in self.by_version.items():
            sources.sort(key=lambda item: (item.char_start, item.char_end, item.chunk_id))
            ends, right, previous = [], 0, None
            for source in sources:
                if len(source.text) != source.char_end - source.char_start:
                    raise ValueError('source_scope_text_extent_invalid')
                if previous is not None and source.char_start < previous.char_end:
                    overlap_end = min(source.char_end, previous.char_end)
                    if source.text[:overlap_end-source.char_start] != previous.text[source.char_start-previous.char_start:overlap_end-previous.char_start]:
                        raise ValueError('source_scope_overlapping_text_disagrees')
                if previous is None or source.char_end > previous.char_end:
                    previous = source
                right = max(right, source.char_end)
                ends.append(right)
            self.ends[version] = ends
        self.available = _merge_scope_intervals((corpus.knowledge_base_id, source.document_version_id,
            source.char_start, source.char_end) for source in corpus.sources)
        self.by_id = {node.id: node for node in self.nodes}
        document_ids_by_version={version:{source.document_id for source in sources} for version,sources in self.by_version.items()}
        self.node_ranges = {}
        self.text = {}
        self.cached_characters = 0
        self.opening_headings = defaultdict(list)
        self.declared_headings = defaultdict(list)
        for node in self.nodes:
            raise_if_source_io_cancelled()
            if (node.knowledge_base_id != corpus.knowledge_base_id or node.document_version_id not in self.by_version
                    or node.document_id not in document_ids_by_version[node.document_version_id]):
                raise ValueError('source_scope_structure_identity_invalid')
            if type(node.char_start) is not int or type(node.char_end) is not int or not 0 <= node.char_start < node.char_end:
                continue
            self.node_ranges[node.id] = (corpus.knowledge_base_id, node.document_version_id, node.char_start, node.char_end)
            parent = self.by_id.get(node.parent_id)
            if node.node_type == 'section':
                self.declared_headings[node.document_version_id].append(node)
            if node.node_type == 'section' and parent is not None and parent.node_type == 'document' and node.char_start == parent.char_start:
                self.opening_headings[parent.id].append(node)
        self.identity = control_hash({'protocol': PROTOCOL, 'source_scope': corpus.scope_hash,
            'sources': [control_hash((source.chunk_id,source.document_id,source.document_version_id,source.title,
                source.char_start,source.char_end,source.text_hash,control_hash(source.text)))
                for source in sorted(corpus.sources,key=lambda item:item.chunk_id)], 'nodes': [
            control_hash({name: getattr(node, name) for name in ('id', 'document_id', 'document_version_id', 'node_type',
                'parent_id', 'previous_sibling_id', 'next_sibling_id', 'title', 'char_start', 'char_end', 'layout_json')})
            for node in sorted(self.nodes, key=lambda n: n.id)]})
        self.structural_identity = self.identity
        self.semantic_selection = None
        self.semantic_matches = {}
        self.role_exclusions=[]
        for node in self.nodes:
            span=self.node_ranges.get(node.id)
            if node.node_type!='section' or span is None or normalized_role_title(node.title) not in SUMMARY_TITLES|CONTENTS_TITLES:
                continue
            raw=self.raw_slice(span[1],span[2],min(span[3],span[2]+512))
            if raw is not None and _title_matches(node.title or '',raw):
                self.role_exclusions.append((span,node.id))

    def selector_ranges(self,selector,node):
        span=self.node_ranges[node.id]
        if selector.match=='role' and selector.role=='detail':
            return _missing_scope_intervals((span,),[part for part,_ in self.role_exclusions])
        return (span,)

    @classmethod
    def load(cls, db, *, corpus, for_update=False, task=None):
        versions = sorted({source.document_version_id for source in corpus.sources})
        types = set(_NODE_KINDS)
        if task is not None:
            requested = set()
            stack = [item.source_scope for item in task.requirements if item.source_scope is not None]
            while stack:
                item = stack.pop()
                stack.extend(item.children)
                if getattr(item,'scope',None) is not None:
                    stack.append(item.scope)
                if getattr(item,'selector',None) is not None:
                    requested.add(item.selector.kind)
            types = {'document','section','caption'} | {key for key,value in _NODE_KINDS.items() if value in requested}
        statement = select(ChunkStructureNode).where(
            ChunkStructureNode.knowledge_base_id == corpus.knowledge_base_id,
            ChunkStructureNode.document_version_id.in_(versions),
            ChunkStructureNode.node_type.in_(types)).order_by(ChunkStructureNode.id).limit(MAX_NODES + 1)
        if for_update:
            statement = statement.with_for_update().execution_options(populate_existing=True)
        nodes = list(db.scalars(statement))
        by_id = {node.id:node for node in nodes}
        pending = {node.previous_sibling_id:node for node in nodes
            if node.node_type in {'table','formula','code_block','figure'} and node.previous_sibling_id not in by_id and node.previous_sibling_id}
        for _ in range(64):
            if not pending or len(nodes)>MAX_NODES:
                break
            raise_if_source_io_cancelled()
            identifiers = sorted(pending)
            following = {}
            for offset in range(0,len(identifiers),256):
                query = select(ChunkStructureNode).where(ChunkStructureNode.id.in_(identifiers[offset:offset+256]),
                    ChunkStructureNode.knowledge_base_id==corpus.knowledge_base_id,
                    ChunkStructureNode.document_version_id.in_(versions),ChunkStructureNode.node_type.in_(_NODE_KINDS))
                if for_update:
                    query=query.with_for_update().execution_options(populate_existing=True)
                for node in db.scalars(query):
                    origin=pending[node.id]
                    if node.id not in by_id:
                        nodes.append(node)
                        by_id[node.id]=node
                    if (node.node_type!='caption' and node.parent_id==origin.parent_id
                        and node.char_start is not None and node.char_end is not None
                        and origin.char_start is not None and origin.char_end is not None
                        and origin.char_start<=node.char_start<node.char_end<=origin.char_end
                        and node.previous_sibling_id and node.previous_sibling_id not in by_id):
                        following[node.previous_sibling_id]=origin
            pending=following
        return cls(corpus=corpus, nodes=nodes, task_hash=task.identity if task is not None else None)

    def raw_slice(self, version, start, end):
        if not 0 <= start <= end or end - start > MAX_TEXT_WINDOW:
            raise ValueError('source_scope_text_window_capacity_exceeded')
        key = version, start, end
        if key in self.text:
            return self.text[key]
        sources = self.by_version.get(version, ())
        pos = bisect_left(self.ends.get(version, ()), start + 1)
        cursor, pieces = start, []
        for source in sources[pos:]:
            if source.char_start >= end:
                break
            if source.char_end <= cursor:
                continue
            if source.char_start > cursor or len(source.text) != source.char_end - source.char_start:
                return None
            right = min(end, source.char_end)
            pieces.append(source.text[cursor-source.char_start:right-source.char_start])
            cursor = right
            if cursor == end:
                break
        value = ''.join(pieces) if cursor == end else None
        if value is None or self.cached_characters + len(value) <= MAX_CACHED_CHARACTERS:
            self.text[key] = value
            self.cached_characters += len(value) if value else 0
        return value

    def _labels(self, node):
        span = self.node_ranges.get(node.id)
        if span is None:
            return ()
        raw = self.raw_slice(span[1], span[2], min(span[3], span[2] + 512))
        if raw is None:
            return ()
        result = []
        kind = _NODE_KINDS[node.node_type]
        def declared(value):
            return (node.node_type == 'section' or re.match(r'^(?:' + _KIND_WORDS[kind] + r')\s*(?=\d|[IVXLCDM]+\b)',
                unicodedata.normalize('NFKC', value).strip(), re.I) is not None)
        if declared(raw) and _label(raw):
            result.append((_label(raw), (node.id,)))
        # A leading caption is a declaration only when the parsed sibling
        # relation and an empty raw-text gap agree. Proximity edges are unused.
        previous = self.by_id.get(node.previous_sibling_id)
        following, seen = node, {node.id}
        while previous is not None and previous.node_type != 'caption':
            bounds = self.node_ranges.get(previous.id)
            if (len(seen) >= 64 or previous.id in seen or not bounds or bounds[:2] != span[:2]
                    or not span[2] <= bounds[2] < bounds[3] <= span[3]
                    or previous.next_sibling_id != following.id or previous.parent_id != node.parent_id):
                previous = None
                break
            seen.add(previous.id)
            following, previous = previous, self.by_id.get(previous.previous_sibling_id)
        if (previous is not None and previous.node_type == 'caption'
                and previous.next_sibling_id == following.id and previous.parent_id == node.parent_id
                and previous.document_version_id == node.document_version_id):
            bounds = self.node_ranges.get(previous.id)
            if bounds and 0 <= span[2] - bounds[3] <= 64:
                gap = self.raw_slice(span[1], bounds[3], span[2]) if bounds[3] < span[2] else ''
                caption = self.raw_slice(span[1], bounds[2], min(bounds[3],bounds[2]+512))
                if gap is not None and not gap.strip() and caption and declared(caption) and _label(caption):
                    result.append((_label(caption), tuple(sorted({previous.id,*seen}))))
        return tuple(result)

    def _matches(self, selector, semantic_key=None):
        if semantic_key in self.semantic_matches:
            node=self.by_id[self.semantic_matches[semantic_key]]
            return ((node,(node.id,)),)
        result = []
        for node in self.nodes:
            if _NODE_KINDS[node.node_type] != selector.kind or node.id not in self.node_ranges:
                continue
            witnesses = (node.id,)
            if selector.match == 'kind':
                matched = True
            elif selector.match=='role':
                matched=selector.role in structure_roles((node,))
                if matched and selector.role=='summary':
                    span=self.node_ranges[node.id]
                    raw=self.raw_slice(span[1],span[2],min(span[3],span[2]+512))
                    matched=raw is not None and _title_matches(node.title or '',raw)
                if matched and selector.role=='detail':
                    witnesses=tuple(sorted({node.id,*[identifier for part,identifier in self.role_exclusions
                        if _intersect_scope_intervals((self.node_ranges[node.id],),(part,))]}))
                    matched=bool(self.selector_ranges(selector,node))
            elif selector.match == 'label':
                witnesses = next((ids for label, ids in self._labels(node) if label == _label(selector.reference)), ())
                matched = bool(witnesses)
            else:
                wanted = _normal(selector.reference)
                matched = _title_matches(wanted,node.title or '')
                if not matched and node.node_type == 'document':
                    matched = _document_alias_matches(wanted, node.title or '')
                if not matched and node.node_type == 'document':
                    # File names and displayed titles can differ. Accept a
                    # parser-declared opening heading, not a mention in prose.
                    headings = self.opening_headings.get(node.id, ())
                    for heading in headings:
                        bounds = self.node_ranges.get(heading.id)
                        raw = self.raw_slice(bounds[1],bounds[2],min(bounds[3],bounds[2]+512)) if bounds else None
                        if (_title_matches(wanted,heading.title or '') and raw is not None
                                and _title_matches(wanted,raw)):
                            matched, witnesses = True,(node.id,heading.id)
                            break
                if not matched and node.node_type == 'document':
                    # Imported bundles may preserve a user-visible document
                    # or guide title as a nested declared heading. Bind that
                    # heading to its owning document version only after both
                    # the parsed title and the raw heading text agree.
                    for heading in self.declared_headings.get(
                        node.document_version_id, ()
                    ):
                        if (
                            heading.document_id == node.document_id
                            and heading.id in self.node_ranges
                            and _title_matches(wanted, heading.title or '')
                        ):
                            matched, witnesses = True, (node.id, heading.id)
                            break
                if matched and node.node_type != 'document':
                    span = self.node_ranges[node.id]
                    raw = self.raw_slice(span[1], span[2], min(span[3], span[2] + 512))
                    matched = raw is not None and _title_matches(wanted,raw)
            if matched:
                result.append((node, witnesses))
        return tuple(result)

    def resolve(self, request):
        candidates = {}
        request_hash=control_hash(request.model_dump(mode='json'))
        def allowed(node,path=()):
            raise_if_source_io_cancelled()
            if node.op == 'scope':
                key = path
                if key not in candidates:
                    semantic_key=control_hash({'request':request_hash,'path':path,'selector':node.selector.model_dump(mode='json')})
                    candidates[key] = self._matches(node.selector,semantic_key)
                return _merge_scope_intervals(span for item,_ in candidates[key] for span in self.selector_ranges(node.selector,item))
            parts = [allowed(child,(*path,index)) for index,child in enumerate(node.children)]
            result = parts[0]
            for part in parts[1:]:
                result = (_intersect_scope_intervals(result, part) if node.op == 'intersection'
                          else _merge_scope_intervals((*result, *part)))
            return result
        allowed(request)
        reasons, witness_ids = set(), set()
        def resolve_node(node, permitted=None,path=()):
            if node.op == 'scope':
                matches = candidates[path]
                if permitted is not None:
                    matches = tuple((item, ids) for item, ids in matches
                        if _intersect_scope_intervals(self.selector_ranges(node.selector,item), permitted))
                # Duplicate split nodes with the same parser identity and span
                # are one location, but distinct repeated labels stay ambiguous.
                unique = {}
                for item, ids in matches:
                    unique.setdefault(self.node_ranges[item.id], (item, ids))
                matches = tuple(unique.values())
                if not matches:
                    reasons.add('no_verified_match')
                    return ()
                if (
                    len(matches) > 1
                    and node.selector.match not in {'kind','role'}
                    and node.selector.kind != 'document'
                ):
                    reasons.add('ambiguous')
                    return ()
                for item, ids in matches:
                    witness_ids.update(ids)
                    span = self.node_ranges[item.id]
                    metadata = (item.layout_json or {}).get('metadata') or {}
                    if (node.selector.kind == 'figure'):
                        reasons.add('unsupported_representation')
                    elif (metadata.get('truncated') or metadata.get('incomplete')
                          or _missing_scope_intervals(self.selector_ranges(node.selector,item),self.available)):
                        reasons.add('representation_incomplete')
                return _merge_scope_intervals(span for item,_ in matches for span in self.selector_ranges(node.selector,item))
            child_permission = permitted
            if node.op == 'intersection':
                raw = allowed(node,path)
                child_permission = raw if permitted is None else _intersect_scope_intervals(raw, permitted)
            parts = [resolve_node(child,child_permission,(*path,index)) for index,child in enumerate(node.children)]
            result = parts[0]
            for part in parts[1:]:
                result = (_intersect_scope_intervals(result, part) if node.op == 'intersection'
                          else _merge_scope_intervals((*result, *part)))
            return result
        ranges = resolve_node(request)
        key = control_hash(request.model_dump(mode='json'))
        reason = next((r for r in ('unsupported_representation', 'ambiguous', 'no_verified_match', 'representation_incomplete')
                       if r in reasons), 'resolved')
        verified = reason in {'resolved', 'representation_incomplete'} and bool(ranges)
        if reason == 'representation_incomplete':
            ranges = _intersect_scope_intervals(ranges, self.available)
            verified = bool(ranges)
        fact = EvidenceScopeFact(id=key, knowledge_base_id=self.corpus.knowledge_base_id,
            kind=request.selector.kind if request.selector else 'text',
            resolution='verified' if verified else 'unsupported' if reason == 'unsupported_representation' else 'unresolved',
            extent_complete=verified and not reasons,
            intervals=tuple(EvidenceInterval(knowledge_base_id=kb, document_version_id=version, start=start, end=end)
                for kb, version, start, end in ranges) if verified else (),
            witness_ids=tuple(sorted(witness_ids)))
        return SourceScopeBinding(request_hash=key, fact=fact, node_ids=tuple(sorted(witness_ids)),
            source_identity_hash=self.identity, reason=reason)

    def bind(self, task):
        if self.task_hash is not None and self.task_hash != task.identity:
            raise ValueError('source_scope_task_identity_changed')
        result = []
        for facet in task.requirements:
            if facet.source_scope is None:
                continue
            validate_scope_declarations(facet.source_scope)
            bindings = {control_hash(item.scope.model_dump(mode='json')): self.resolve(item.scope)
                        for item in coverage_requests(facet.source_scope)}
            result.append(FacetScopeInput(facet_id=facet.id, bindings=tuple(bindings[key] for key in sorted(bindings))))
        return tuple(result)


def scope_target_plan(*, index, task, token_budget, target_limit, candidate_ids=None,
                      affinities=None, overlap_target_budget=1):
    """Propose source addresses to the existing graph executor, never raw hits."""
    from app.retrieval_control_contracts import EvidenceIntervalCandidate
    from app.services.chunking import rough_token_count
    from app.services.context_packing import plan_interval_scope_completion
    if (token_budget <= 0 or not 0 <= target_limit <= 256
            or type(overlap_target_budget) is not int
            or not 1 <= overlap_target_budget <= 8):
        raise ValueError('source_scope_target_budget_invalid')
    if candidate_ids is not None and not set(candidate_ids).issubset(index.corpus.by_id):
        raise ValueError('source_scope_target_candidate_identity_invalid')
    if affinities is not None:
        if set(affinities)!={facet.id for facet in task.requirements}:
            raise ValueError('scope_target_affinity_facet_mismatch')
        for values in affinities.values():
            if set(values)!=set(index.corpus.by_id) or any(not math.isfinite(value) or not 0<=value<=1 for value in values.values()):
                raise ValueError('scope_target_affinity_source_mismatch')
    bound = index.bind(task)
    plans, targets, packing_targets = [], set(), []
    for facet in task.requirements:
        if facet.source_scope is None:
            continue
        binding = next(item for item in bound if item.facet_id == facet.id)
        by_request = {item.request_hash: item for item in binding.bindings}
        def propose(obligation):
            raise_if_source_io_cancelled()
            if obligation.op != 'coverage':
                choices = [propose(child) for child in obligation.children]
                if obligation.op == 'any':
                    feasible = [item for item in choices if item is not None]
                    return min(feasible,key=lambda ids:(sum(costs[cid] for cid in ids),len(ids),tuple(sorted(ids)))) if feasible else None
                return set().union(*choices) if all(item is not None for item in choices) else None
            item = by_request[control_hash(obligation.scope.model_dump(mode='json'))]
            if item.reason != 'resolved' and not (obligation.mode == 'overlap' and item.fact.resolution == 'verified'):
                return None
            coverage = evaluate_evidence_scope(knowledge_base_id=task.knowledge_base_id,
                expression=EvidenceScopeExpression(op='scope',scope_id=item.fact.id),scopes=(item.fact,),packed=(),mode='complete')
            candidates = []
            spans = _merge_scope_intervals((part.knowledge_base_id,part.document_version_id,part.start,part.end) for part in item.fact.intervals)
            for source in index.corpus.sources:
                if candidate_ids is not None and source.chunk_id not in candidate_ids:
                    continue
                interval = (task.knowledge_base_id,source.document_version_id,source.char_start,source.char_end)
                if _intersect_scope_intervals((interval,),spans):
                    local_witnesses=tuple(witness for witness in item.fact.witness_ids
                        if witness in index.node_ranges and _intersect_scope_intervals((interval,),(index.node_ranges[witness],)))
                    if not local_witnesses:
                        raise ValueError('scope_target_local_witness_missing')
                    candidates.append(EvidenceIntervalCandidate(id=source.chunk_id,
                        interval=EvidenceInterval(knowledge_base_id=task.knowledge_base_id,document_version_id=source.document_version_id,
                            start=source.char_start,end=source.char_end),cost=costs[source.chunk_id],witness_ids=local_witnesses))
            if obligation.mode == 'overlap':
                # Overlap never materializes the whole object. The target
                # executor may request a small bounded set when one facet asks
                # for several co-located facts in the same verified scope.
                feasible=[candidate for candidate in candidates if candidate.cost<=token_budget]
                ordered=sorted(feasible,key=lambda c:(-affinities[facet.id][c.id],c.cost,c.id)
                    if affinities is not None else (c.cost,c.id))
                chosen=[]
                total=0
                for candidate in ordered:
                    if len(chosen)>=overlap_target_budget or total+candidate.cost>token_budget:
                        continue
                    chosen.append(candidate.id)
                    total+=candidate.cost
                packing_targets.extend(chosen)
                return set(chosen) if chosen else None
            plan = plan_interval_scope_completion(knowledge_base_id=task.knowledge_base_id,coverage=coverage,
                candidates=candidates,budget=token_budget)
            plans.append(plan.model_dump(mode='json'))
            packing_targets.extend(plan.selected_ids)
            return set(plan.selected_ids) if plan.status in {'ready','already_covered'} else None
        costs = {source.chunk_id:rough_token_count(source.text) for source in index.corpus.sources}
        chosen = propose(facet.source_scope)
        if chosen is not None:
            targets.update(chosen)
    cost = sum(costs[cid] for cid in targets) if targets else 0
    if len(targets) > target_limit or cost > token_budget:
        status, targets, packing_targets = 'over_budget', set(), []
    else:
        status = 'proposed' if targets else 'no_resolved_target'
    packing_target_ids = list(dict.fromkeys(
        [item for item in packing_targets if item in targets]
        + sorted(targets - set(packing_targets))
    ))
    record = {'protocol_version':'source_scope_target_plan_v1','task_hash':task.identity,
        'source_identity_hash':index.identity,'target_chunk_ids':sorted(targets),'status':status,
        'packing_target_chunk_ids':packing_target_ids,
        'token_cost':cost,'token_budget':token_budget,'target_limit':target_limit,'interval_plans':plans,
        'overlap_target_budget':overlap_target_budget,
        'model_call_count':0,'executor_validation_required':True,
        'cross_obligation_global_optimum_claimed':False}
    if affinities is not None:
        record.update(entry_selection_protocol='scope_affinity_entry_v1',affinity_input_hash=control_hash(affinities))
    record['audit_hash'] = control_hash(record)
    return tuple(packing_target_ids), record


def evaluate_scope_obligation(*, task, facet, bound, packed):
    if facet.source_scope is None or bound.facet_id != facet.id:
        raise ValueError('source_scope_obligation_identity_invalid')
    by_request = {item.request_hash: item for item in bound.bindings}
    coverage, reasons = [], set()
    def evaluate(node):
        if node.op != 'coverage':
            return combine_scope_states([evaluate(child) for child in node.children], op=node.op)
        key = control_hash(node.scope.model_dump(mode='json'))
        binding = by_request[key]
        result = evaluate_evidence_scope(knowledge_base_id=task.knowledge_base_id,
            expression=EvidenceScopeExpression(op='scope', scope_id=binding.fact.id),
            scopes=(binding.fact,), packed=packed, mode=node.mode)
        coverage.append(result)
        if binding.reason != 'resolved':
            reasons.add(binding.reason)
        # A verified readable part can satisfy overlap. Only a request for the
        # whole object requires proof that its representation is complete.
        unresolved = binding.reason != 'resolved' and not (
            node.mode == 'overlap' and binding.fact.resolution == 'verified')
        return 'unknown' if unresolved else result.state
    state = evaluate(facet.source_scope)
    return FacetScopeStatus(facet_id=facet.id, state=state, coverage=tuple(coverage), reason_codes=tuple(sorted(reasons)),
        input_hash=control_hash({'task': task.identity, 'facet': facet.id, 'binding': bound.model_dump(mode='json'),
                                'packed': [item.model_dump(mode='json') for item in packed]}))


def package_scope_intervals(task, package):
    return tuple(EvidenceInterval(knowledge_base_id=task.knowledge_base_id,
        document_version_id=item['document_version_id'], start=item['char_span'][0], end=item['char_span'][1])
        for item in package.package_json['chunks'])


def resolved_scope_filters(*,task,index,filters):
    if index is None or any(facet.source_scope is None for facet in task.requirements):
        return filters,None
    bindings=index.bind(task)
    if any(binding.fact.resolution!='verified' or not binding.fact.intervals for group in bindings for binding in group.bindings):
        return filters,None
    versions={part.document_version_id for group in bindings for binding in group.bindings for part in binding.fact.intervals}
    documents={source.document_id for source in index.corpus.sources if source.document_version_id in versions}
    if filters.document_ids:
        documents &= set(filters.document_ids)
    if not documents:
        raise ValueError('source_scope_effective_filter_empty')
    effective=filters.model_copy(update={'document_ids':sorted(documents)})
    audit={'protocol_version':'resolved_scope_filter_v1','task_hash':task.identity,'source_identity_hash':index.identity,
        'original_filters':filters.model_dump(mode='json'),'effective_filters':effective.model_dump(mode='json'),
        'scope_binding_hash':control_hash([group.model_dump(mode='json') for group in bindings]),
        'document_count':len(documents),'model_call_count':0}
    audit['audit_hash']=control_hash(audit)
    return effective,audit


def generation_scope_guidance(*, task, parameters, evidence):
    """Expose usable text offsets with opaque handles; no database addresses."""
    result = []
    for item in parameters.scope_inputs:
        ranges = _merge_scope_intervals((part.knowledge_base_id,part.document_version_id,part.start,part.end)
            for binding in item.bindings if binding.fact.resolution == 'verified' for part in binding.fact.intervals)
        sources = []
        for source in evidence.sources:
            span = source['source_span']
            start,end = span['char_span']
            usable = _intersect_scope_intervals(((task.knowledge_base_id,span['document_version_id'],start,end),),ranges)
            if usable:
                sources.append({'source_handle':source['source_handle'],
                    'text_char_spans':[[left-start,right-start] for _,_,left,right in usable]})
        result.append({'requirement_id':item.facet_id,'sources':sources})
    from app.retrieval_control_contracts import GenerationSourceScopeGuidance
    return GenerationSourceScopeGuidance(requirements=result) if result else None


def load_scope_replay_index(db,*,task,ids,source_scope_hash,for_update=False,task_scoped=True):
    from types import SimpleNamespace
    from app.models import Chunk, Document
    from app.services.retrieval_corpus import CorpusSource
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('source_scope_replay_source_identity_invalid')
    statement = select(Chunk,Document.title).join(Document,Document.id == Chunk.document_id).where(
        Chunk.id.in_(ids),Chunk.knowledge_base_id == task.knowledge_base_id).order_by(Chunk.id)
    if for_update:
        statement = statement.with_for_update().execution_options(populate_existing=True)
    rows = list(db.execute(statement))
    if len(rows) != len(ids):
        raise ValueError('source_scope_replay_source_missing')
    sources = tuple(CorpusSource(chunk.id,chunk.document_id,chunk.document_version_id,title,chunk.text,
        chunk.char_start,chunk.char_end,chunk.text_hash,chunk.section_path) for chunk,title in rows)
    corpus = SimpleNamespace(knowledge_base_id=task.knowledge_base_id,sources=sources,
        scope_hash=source_scope_hash,by_id={source.chunk_id:source for source in sources})
    return StructureScopeIndex.load(db,corpus=corpus,for_update=for_update,task=task if task_scoped else None)


def replay_scope_inputs(db, *, task, parameters, package, for_update=False):
    """Rebind persisted control facts to DB sources; no model or graph search."""
    if not parameters.scope_inputs:
        return
    index=load_scope_replay_index(db,task=task,ids=parameters.scope_source_chunk_ids,
        source_scope_hash=parameters.identity.source_scope_hash,for_update=for_update,
        task_scoped=parameters.scope_index_protocol=='task_structure_index_v1')
    if parameters.scope_selection is not None:
        from app.services.source_location import replay_scope_selection
        replay_scope_selection(db,task=task,index=index,selection=parameters.scope_selection,for_update=for_update)
    if index.bind(task) != parameters.scope_inputs:
        raise ValueError('source_scope_replay_binding_changed')
    if package_scope_intervals(task,package) != parameters.packed_scope_intervals:
        raise ValueError('source_scope_replay_packed_span_changed')
