import re
from collections import namedtuple
import unicodedata

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


def text_preprocessing(chunk, word):
    """
    Creates inputs for LLMs. Returns text chunk with word appended at the beginning. 
    :param chunk: str
    :param word: str
    :return: str
    """
    preproc_text = f'Word used: {word}' + ' [...] ' + re.sub(r'  +', ' ',
                                                             unicodedata.normalize('NFKD', chunk.strip().strip(
                                                                 '.'))).strip() + ' [...]'
    return preproc_text
