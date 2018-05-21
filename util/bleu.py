import argparse
from typing import List, Dict
import nltk
import numpy as np


def brevity_penalty(reference: List, hypothesis: List) -> float:
    return min(1, np.exp(1 - len(reference) / len(hypothesis)))


def ngrams(words: List[str], n: int) -> Dict[str, int]:
    """
    Count word level n-grams for a given list of words and an n-gram order n.
    """
    counts = {}
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])
        counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def modified_precision(reference: List[str],
                       hypothesis: List[str],
                       n: int) -> float:
    """
    Compute the modified precision of a reference sentence and a hypothesis
    sentence for a given order of n-grams n.
    """
    reference_ngrams = ngrams(reference, n)
    hypothesis_ngrams = ngrams(hypothesis, n)

    overlapping_ngrams = reference_ngrams.keys() & hypothesis_ngrams.keys()

    clipped_counts = {}
    for ngram in overlapping_ngrams:
        clipped_counts[ngram] = min(hypothesis_ngrams[ngram], reference_ngrams[ngram])

    correct = sum(clipped_counts.values())
    hyp_length = sum(hypothesis_ngrams.values())

    return correct / hyp_length


def prepare(file: str) -> List[str]:
    """
    Prepare a file-like object for scoring
    :param file: the path to the file
    :return: All tokens of the text in a list of strings
    """
    with open(file) as f:
        return nltk.word_tokenize(f.read())


def bleu_score(reference: List[str],
               hypothesis: List[str],
               n_max: int = 4) -> float:
    """
    Compute the bleu-score for a hypothesis with a single reference.
    """

    # Next we compute the modified precision for all n in [1,n_max]
    p_n = np.array([modified_precision(reference, hypothesis, n) for n in
                    range(1, n_max + 1)])

    # We set the weight to 1 for all n
    weight_n = np.array([1] * n_max)

    # We compute the product of all weighted precisions to the power of 1/n_max
    p = np.prod(weight_n * p_n)
    p = pow(p, 1 / n_max)

    # We compute the brevity penalty
    bp = brevity_penalty(reference, hypothesis)

    return bp * p


def get_argument_parser():
    argument_parser = argparse.ArgumentParser(description='Run an xml-rpc moses-style mt-server')
    argument_parser.add_argument('reference',
                                 help = "a file of reference translations")
    argument_parser.add_argument('hypothesis',
                                 help = "a file of mt-translations"
                                 )
    argument_parser.add_argument('-n', type=int, default=4,
                                 help="Max n-grams")

    return argument_parser


def main():
    args = get_argument_parser().parse_args()

    reference = prepare(args.reference)
    hypothesis = prepare(args.hypothesis)

    print(bleu_score(reference, hypothesis, args.n))

if __name__ == "__main__":

    main()
