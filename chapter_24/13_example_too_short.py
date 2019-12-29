# very short
from nltk.translate.bleu_score import sentence_bleu
import warnings
warnings.simplefilter("ignore")

reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick']
score = sentence_bleu(reference, candidate)
print(score)