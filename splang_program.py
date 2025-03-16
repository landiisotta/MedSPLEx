import dspy
from typing import Literal
import yaml
import logging


# Stigmatizing and Privileging Language Classification Signature
class SPLangClassification(dspy.Signature):
    """Classify whether the word indicated at the beginning of the chunk extracted from a clinical note carries a privileging, stigmatizing, or neutral valence in the context provided."""

    word: str = dspy.InputField()
    chunk: str = dspy.InputField()
    valence: Literal["privileging", "stigmatizing", "neutral"] = dspy.OutputField()


# Program reading in a text chunk and returning the valence either using
# keyword matching labels or a LM.
class SPLangProgram(dspy.Module):
    def __init__(self):
        self.label = dspy.Predict(SPLangClassification)
        self.valence_dict = yaml.safe_load(open('config/keywords.yaml'))

    def forward(self, word, chunk):
        try:
            word_usage = self.valence_dict[word]['valence']
        except KeyError:
            logging.ERROR("Word valence not found, please update the keywords file.")
            return None
        if word_usage == 'ambiguous':
            return self.label(word=word, chunk=chunk)
        else:
            return dspy.Prediction(valence=self.valence_dict[word]['valence'])


# Program reading in a text chunk and returning the valence either using
# keyword matching labels or a LM.
class F1SPLangProgram(dspy.Module):
    def __init__(self):
        self.label = dspy.Predict(SPLangClassification)
        self.valence_dict = yaml.safe_load(open('config/keywords.yaml'))

    def forward(self, words, chunks):
        preds = []
        for word, chunk in zip(words, chunks):
            try:
                word_usage = self.valence_dict[word]
            except KeyError:
                logging.ERROR("Word valence not found, please update the keywords file.")
                return
            if word_usage == 'ambiguous':
                preds.append(self.label(word=word, chunk=chunk).valence)
            else:
                preds.append(self.valence_dict[word]['valence'])
        return dspy.Prediction(valences=preds)
