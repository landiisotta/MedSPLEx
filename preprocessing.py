import re
from collections import namedtuple

Chunk = namedtuple('Chunk', ['keyword', 'text'])
def chunkizer(text: str, config: dict, nchar=200) -> list[tuple]:
    """
    Returns a list of tuples where each tuple is a chunk of text and associate keyword.
    """
    chunks = []
    for keyword in config.keys():
        for match in re.finditer(keyword['regex'], text, re.IGNORECASE):
            start = max(0, match.start() - nchar)
            end = min(len(text), match.end() + nchar)
            chunk = text[start:end]
            chunks.append(Chunk(keyword=keyword, text=chunk))
    return chunks
