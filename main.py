
def count_occurrences(corpus):
    """
    This function counts the occurrences of each
        tag,
        word, tag
        tag, tag
    and saves it on a file if it doens't exist (if file already exists, it is not saved)
    :param corpus: all the corpus
    :return: returns a json with the counts ({tags:{...}, ...})
    """
    pass


def calculate_emission_probs(tagprobs, wordtagprobs):
    """
    Calculate the emission probabilities and saves it on a file if the file doesn't exist
    :param tagprobs: json of the probabilities of the tags
    :param wordtagprobs: json of probabilities of a word given the tag
    :return: emission probabilites
    """
    pass


def calculate_transition_probs(tagprobs, tagtagprobs):
    """
    Calculates the transition probs and saves it on a file if the file doesn't exist
    :param tagprobs: json of prob of tags
    :param tagtagprobs: json of prob of tagtag
    :return: transition probabilities for that tag
    """
    pass


def main():
    pass


if __name__ == "__main__":
    main()
