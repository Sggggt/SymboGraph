"""Separate immutable user mentions from identifiers that need literal witnesses."""
import re


def formal_literal(value: str) -> bool:
    if not value or not value.isascii():
        return False
    return bool(re.fullmatch(r"(?:[A-Z]{2,}[A-Z0-9_-]*|[A-Za-z]+[0-9][A-Za-z0-9_-]*|[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+|[0-9]{1,8}(?:\.[0-9]+)?)", value))


def formal_literals_match(requirement, title: str, text: str) -> bool:
    content = title + '\n' + text
    return all(re.search(r'(?<!\w)' + re.escape(value) + r'(?!\w)', content, re.I)
               for value in requirement.protected_literals if formal_literal(value))
