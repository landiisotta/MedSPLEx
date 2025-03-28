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


def text_preprocessing(chunk, word, text_col='text_preproc', config=None):
    """
    Creates inputs for LLMs. Returns text chunk with word appended at the beginning. 
    :param chunk: str
    :param word: str
    :param config: dict
    :return: str
    """
    if config:
        word_config = config[word]
        word_stigmatizing_score = word_config['stigmatizing_score']
        word_privileging_score = word_config['privileging_score']
        if word_stigmatizing_score == 5:
            range_text = 'privileging or neutral'
        elif word_privileging_score == 5:
            range_text = 'stigmatizing or neutral'
        else:
            range_text = 'privileging, stigmatizing, or neutral'

        explanation = f""" This word can have a {range_text} meaning."""
    else:
        explanation = ''
    if text_col == 'text_preproc':
        preproc_text = f'Word used: {word}.' + explanation + ' [...] ' + re.sub(r'  +', ' ',
                                                                                unicodedata.normalize('NFKD',
                                                                                                      chunk.strip().strip(
                                                                                                          '.'))).strip() + ' [...]'
    else:
        preproc_text = explanation + re.sub(r'  +', ' ',
                                            unicodedata.normalize('NFKD',
                                                                  chunk.strip().strip(
                                                                      '.'))).strip()
    return preproc_text
