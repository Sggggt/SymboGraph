"""Shared structural role vocabulary; no document-specific names or answer terms."""

SUMMARY_TITLES=frozenset({'abstract','summary','executive summary','摘要','概述'})
DETAIL_TITLES=frozenset({'detail','details','detailed description','detailed discussion','详细','详细说明','正文'})
CONTENTS_TITLES=frozenset({'contents','table of contents','目录'})


def normalized_role_title(title):
    return str(title or '').strip('# \n\t').casefold()


def structure_roles(nodes):
    titles={normalized_role_title(node.title) for node in nodes}
    types={str(node.node_type or '') for node in nodes}
    summary=bool(titles&SUMMARY_TITLES)
    contents=bool(titles&CONTENTS_TITLES)
    roles=set()
    if summary:
        roles.add('summary')
    if titles and not summary and not contents:
        roles.add('detail')
    roles.update(role for kind,role in (('table','table'),('formula','formula'),('code_block','code')) if kind in types)
    return roles
