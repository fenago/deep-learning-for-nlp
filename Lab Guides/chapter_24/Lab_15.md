# How to Evaluate Generated Text With the BLEU Score
BLEU, or the Bilingual Evaluation Understudy, is a score for comparing a candidate translation
of text to one or more reference translations. Although developed for translation, it can be used
to evaluate text generated for a suite of natural language processing tasks. In this tutorial, you
will discover the BLEU score for evaluating and scoring candidate text using the NLTK library
in Python. After completing this tutorial, you will know:
- A gentle introduction to the BLEU score and an intuition for what is being calculated.
- How you can calculate BLEU scores in Python using the NLTK library for sentences and
documents.
- How you can use a suite of small examples to develop an intuition for how differences
between a candidate and reference text impact the final BLEU score.

Let's get started.

24.1

Tutorial Overview

This tutorial is divided into the following parts:
1. Bilingual Evaluation Understudy Score
2. Calculate BLEU Scores
3. Cumulative and Individual BLEU Scores
4. Worked Examples

24.2

Bilingual Evaluation Understudy Score

The Bilingual Evaluation Understudy Score, or BLEU for short, is a metric for evaluating a
generated sentence to a reference sentence. A perfect match results in a score of 1.0, whereas a
perfect mismatch results in a score of 0.0. The score was developed for evaluating the predictions
267

24.2. Bilingual Evaluation Understudy Score

268

made by automatic machine translation systems. It is not perfect, but does offer 5 compelling
benefits:
- It is quick and inexpensive to calculate.
- It is easy to understand.
- It is language independent.
- It correlates highly with human evaluation.
- It has been widely adopted.

The BLEU score was proposed by Kishore Papineni, et al. in their 2002 paper BLEU: a
Method for Automatic Evaluation of Machine Translation. The approach works by counting
matching n-grams in the candidate translation to n-grams in the reference text, where 1-gram
or unigram would be each token and a bigram comparison would be each word pair. The
comparison is made regardless of word order.
The primary programming task for a BLEU implementor is to compare n-grams of
the candidate with the n-grams of the reference translation and count the number
of matches. These matches are position-independent. The more the matches, the
better the candidate translation is.
— BLEU: a Method for Automatic Evaluation of Machine Translation, 2002.
The counting of matching n-grams is modified to ensure that it takes the occurrence of the
words in the reference text into account, not rewarding a candidate translation that generates
an abundance of reasonable words. This is referred to in the paper as modified n-gram precision.
Unfortunately, MT systems can overgenerate “reasonable” words, resulting in improbable, but high-precision, translations [...] Intuitively the problem is clear: a
reference word should be considered exhausted after a matching candidate word is
identified. We formalize this intuition as the modified unigram precision.
— BLEU: a Method for Automatic Evaluation of Machine Translation, 2002.
The score is for comparing sentences, but a modified version that normalizes n-grams by
their occurrence is also proposed for better scoring blocks of multiple sentences.
We first compute the n-gram matches sentence by sentence. Next, we add the clipped
n-gram counts for all the candidate sentences and divide by the number of candidate
n-grams in the test corpus to compute a modified precision score, pn, for the entire
test corpus.
— BLEU: a Method for Automatic Evaluation of Machine Translation, 2002.
A perfect score is not possible in practice as a translation would have to match the reference
exactly. This is not even possible by human translators. The number and quality of the
references used to calculate the BLEU score means that comparing scores across datasets can
be troublesome.

24.3. Calculate BLEU Scores

269

The BLEU metric ranges from 0 to 1. Few translations will attain a score of 1
unless they are identical to a reference translation. For this reason, even a human
translator will not necessarily score 1. [...] on a test corpus of about 500 sentences
(40 general news stories), a human translator scored 0.3468 against four references
and scored 0.2571 against two references.
— BLEU: a Method for Automatic Evaluation of Machine Translation, 2002.
In addition to translation, we can use the BLEU score for other language generation problems
with deep learning methods such as:
- Language generation.
- Image caption generation.
- Text summarization.
- Speech recognition.
- And much more.

24.3

Calculate BLEU Scores

The Python Natural Language Toolkit library, or NLTK, provides an implementation of the
BLEU score that you can use to evaluate your generated text against a reference.

24.3.1

Sentence BLEU Score

NLTK provides the sentence bleu() function for evaluating a candidate sentence against one
or more reference sentences. The reference sentences must be provided as a list of sentences
where each reference is a list of tokens. The candidate sentence is provided as a list of tokens.
For example:

```
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate)
print(score)

```

Running this example prints a perfect score as the candidate matches one of the references
exactly.

```
1.0

```


# Corpus BLEU Score

NLTK also provides a function called corpus bleu() for calculating the BLEU score for multiple
sentences such as a paragraph or a document. The references must be specified as a list of
documents where each document is a list of references and each alternative reference is a list of
tokens, e.g. a list of lists of lists of tokens. The candidate documents must be specified as a list
where each document is a list of tokens, e.g. a list of lists of tokens. This is a little confusing;
here is an example of two references for one document.

```
# two references for one document
from nltk.translate.bleu_score import corpus_bleu
references = [[['this', 'is', 'a', 'test'], ['this', 'is' 'test']]]
candidates = [['this', 'is', 'a', 'test']]
score = corpus_bleu(references, candidates)
print(score)

```

Running the example prints a perfect score as before.

```
1.0

```


24.4

Cumulative and Individual BLEU Scores

The BLEU score calculations in NLTK allow you to specify the weighting of different n-grams
in the calculation of the BLEU score. This gives you the flexibility to calculate different types
of BLEU score, such as individual and cumulative n-gram scores. Let's take a look.

24.4.1

Individual n-gram Scores

An individual n-gram score is the evaluation of just matching grams of a specific order, such
as single words (1-gram) or word pairs (2-gram or bigram). The weights are specified as a
tuple where each index refers to the gram order. To calculate the BLEU score only for 1-gram
matches, you can specify a weight of 1 for 1-gram and 0 for 2, 3 and 4 (1, 0, 0, 0). For example:

```
# 1-gram individual BLEU
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
print(score)

```

Running this example prints a score of 0.5.

```
0.75

```

We can repeat this example for individual n-grams from 1 to 4 as follows:

```
# n-gram individual BLEU
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'a', 'test']]
candidate = ['this', 'is', 'a', 'test']
print('Individual 1-gram: %f' % sentence_bleu(reference,
print('Individual 2-gram: %f' % sentence_bleu(reference,
print('Individual 3-gram: %f' % sentence_bleu(reference,
print('Individual 4-gram: %f' % sentence_bleu(reference,

271

candidate,
candidate,
candidate,
candidate,

weights=(1,
weights=(0,
weights=(0,
weights=(0,

0,
1,
0,
0,

0,
0,
1,
0,

0)))
0)))
0)))
1)))

```

Running the example gives the following results.

```
Individual
Individual
Individual
Individual

1-gram:
2-gram:
3-gram:
4-gram:

1.000000
1.000000
1.000000
1.000000

```

Although we can calculate the individual BLEU scores, this is not how the method was
intended to be used and the scores do not carry a lot of meaning, or seem that interpretable.

24.4.2

Cumulative n-gram Scores

Cumulative scores refer to the calculation of individual n-gram scores at all orders from 1 to n and
weighting them by calculating the weighted geometric mean. By default, the sentence bleu()
and corpus bleu() scores calculate the cumulative 4-gram BLEU score, also called BLEU-4.
The weights for the BLEU-4 are 1/4 (25%) or 0.25 for each of the 1-gram, 2-gram, 3-gram and
4-gram scores. For example:

```
# 4-gram cumulative BLEU
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
print(score)

```

Running this example prints the following score:

```
0.707106781187

```

The cumulative and individual 1-gram BLEU use the same weights, e.g. (1, 0, 0, 0). The
2-gram weights assign a 50% to each of 1-gram and 2-gram and the 3-gram weights are 33%
for each of the 1, 2 and 3-gram scores. Let's make this concrete by calculating the cumulative
scores for BLEU-1, BLEU-2, BLEU-3 and BLEU-4:

```
# cumulative BLEU scores
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0,
0)))
print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33,
0.33, 0)))
print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25,
0.25, 0.25)))

```

Running the example prints the following scores. They are quite different and more expressive
than the They are quite different and more expressive than the standalone individual n-gram
scores.

```
Cumulative
Cumulative
Cumulative
Cumulative

1-gram:
2-gram:
3-gram:
4-gram:

0.750000
0.500000
0.632878
0.707107

```

It is common to report the cumulative BLEU-1 to BLEU-4 scores when describing the skill
of a text generation system.

24.5

Worked Examples

In this section, we try to develop further intuition for the BLEU score with some examples. We
work at the sentence level with a single reference sentence of the following:

```
the quick brown fox jumped over the lazy dog

```

First, let's look at a perfect score.

```
# prefect match
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
score = sentence_bleu(reference, candidate)
print(score)

```

Running the example prints a perfect match.

```
1.0

```

Next, let's change one word, 'quick ' to 'fast'.

```
# one word different
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
score = sentence_bleu(reference, candidate)
print(score)

```

This result is a slight drop in score.

```
0.7506238537503395

```

Try changing two words, both 'quick ' to 'fast' and 'lazy' to 'sleepy'.

```
# two words different
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'sleepy', 'dog']
score = sentence_bleu(reference, candidate)
print(score)

```

Running the example, we can see a linear drop in skill.

```
0.4854917717073234

```

What about if all words are different in the candidate?

```
# all words different
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
score = sentence_bleu(reference, candidate)
print(score)

```

We get the worse possible score.

```
0.0

```

Now, let's try a candidate that has fewer words than the reference (e.g. drop the last two
words), but the words are all correct.

```
# shorter candidate
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the']
score = sentence_bleu(reference, candidate)
print(score)

```

The score is much like the score when two words were wrong above.

```
0.7514772930752859

```

How about if we make the candidate two words longer than the reference?

```
# longer candidate
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog',
'from', 'space']
score = sentence_bleu(reference, candidate)
print(score)

```

Again, we can see that our intuition holds and the score is something like two words wrong.

```
0.7860753021519787

```

Finally, let's compare a candidate that is way too short: only two words in length.

```
# very short
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick']
score = sentence_bleu(reference, candidate)
print(score)

```

Running this example first prints a warning message indicating that the 3-gram and above
part of the evaluation (up to 4-gram) cannot be performed. This is fair given we only have
2-grams to work with in the candidate.

```
UserWarning:
Corpus/Sentence contains 0 counts of 3-gram overlaps.
BLEU scores might be undesirable; use SmoothingFunction().
warnings.warn(_msg)

```

Next, we can a score that is very low indeed.

```
0.0301973834223185

```

I encourage you to continue to play with examples. The math is pretty simple and I would
also encourage you to read the paper and explore calculating the sentence-level score yourself in
a spreadsheet.


# Further Reading

This section provides more resources on the topic if you are looking go deeper.
- BLEU on Wikipedia.
https://en.wikipedia.org/wiki/BLEU
- BLEU: a Method for Automatic Evaluation of Machine Translation, 2002.
http://www.aclweb.org/anthology/P02-1040.pdf
- Source code for nltk.translate.bleu score.
http://www.nltk.org/_modules/nltk/translate/bleu_score.html
- nltk.translate package API Documentation.
http://www.nltk.org/api/nltk.translate.html

# Summary

In this tutorial, you discovered the BLEU score for evaluating and scoring candidate text to
reference text in machine translation and other language generation tasks. Specifically, you
learned:
- A gentle introduction to the BLEU score and an intuition for what is being calculated.
- How you can calculate BLEU scores in Python using the NLTK library for sentences and
documents.
- How to can use a suite of small examples to develop an intuition for how differences
between a candidate and reference text impact the final BLEU score.
