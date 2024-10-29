import torch
import transformers

class BaseModel():
    def __init__(self):
        "None"

    def CWI(self, text: str):
        """
        Complex Word Identification 
        Return the most complex word in the text
        """
        first = text.split(' ')[1]
        return first
    
    def SG(self, text: str, word: str):
        """
        Substitute Generation
        Return a list of candidates
        """
        return [word]
    
    def SR(self, text: str, word: str, candidates: list):
        """
        Substitute Ranking
        Return one candidate
        """
        return candidates[0]
