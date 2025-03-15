import dspy
from typing import Literal
import yaml
import logging


# Stigmatizing and Privileging Language Classification Signature
class SPLangClassification(dspy.Signature):
    f"""{yaml.safe_load(open("config/signature_docstring.yaml", 'r'))['docstring']}"""

    word: str = dspy.InputField()
    chunk: str = dspy.InputField()
    valence: Literal["privileging", "stigmatizing", "neutral"] = dspy.OutputField()


# Program reading in a text chunk and returning the valence either using
# keyword matching labels or a LM.
class SPLangProgram(dspy.Module):
    def __init__(self):
        self.label = dspy.Prediction(SPLangClassification)
        self.valence_dict = yaml.safe_load(open('config/word-list.yaml'))

    def forward(self, word, chunk):
        try:
            valence = self.valence_dict[word]
        except KeyError:
            logging.ERROR("Word valence not found, Please update the keyword-valence file.")
            return
        if valence == 'ambiguous':
            return self.label(word=word, chunk=chunk)
        else:
            return self.Predict(valence=self.valence_dict[word]['valence'])
