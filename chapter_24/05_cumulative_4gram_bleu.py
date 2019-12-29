# 4-gram cumulative BLEU
from nltk.translate.bleu_score import sentence_bleu
import warnings
warnings.simplefilter("ignore")

reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
print(score)