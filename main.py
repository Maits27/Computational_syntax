import json, glob, os
from pathlib import Path

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
    total_counts = {'English': {}, 'Spanish': {}}
    output_path = Path('data/counts.json')

    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            total_counts = json.load(f)
    else:
        for lang in total_counts:
            counts = {'transitions': {}, 'emissions': {}, 'tags': {'<BOL>': 1}}
            with open(glob.glob(os.path.join(f'{corpus}/{lang}/', '*train.conllu'))[0], 'r', encoding='utf-8') as f:
                tag = '<BOL>'
                for line in f:
                    prev_tag, word = tag, ''
                    if line == '\n':tag = '<EOL>'
                    elif line[0] == '#': tag = '<BOL>'
                    else: w_id, word, lemma, tag, _, _, _, tag2, _, _ = line.split('\t')

                    if not (tag == '<BOL>' and prev_tag == '<BOL>'):
                        if tag not in counts['tags']: counts['tags'][tag] = 1
                        else: counts['tags'][tag] += 1

                        if not (tag == '<BOL>' and prev_tag == '<EOL>'):
                            if f'{prev_tag}, {tag}' not in counts['transitions']: counts['transitions'][f'{prev_tag}, {tag}'] = 1
                            else: counts['transitions'][f'{prev_tag}, {tag}'] += 1

                        if word != '':
                            if f'{tag}, {word}' not in counts['emissions']: counts['emissions'][f'{tag}, {word}'] = 1
                            else: counts['emissions'][f'{tag}, {word}'] += 1

            total_counts[lang] = counts

        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(total_counts, output_file, indent=4)

    return total_counts



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
    count_occurrences(Path('UD-Data'))


if __name__ == "__main__":
    main()
