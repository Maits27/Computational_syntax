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
                    if line == '\n':
                        tag = '<EOL>'
                    elif line[0] == '#':
                        tag = '<BOL>'
                    else:
                        w_id, word, lemma, tag, _, _, _, tag2, _, _ = line.split('\t')

                    if not (tag == '<BOL>' and prev_tag == '<BOL>'):
                        if tag not in counts['tags']:
                            counts['tags'][tag] = 1
                        else:
                            counts['tags'][tag] += 1

                        if not (tag == '<BOL>' and prev_tag == '<EOL>'):
                            if f'{prev_tag}, {tag}' not in counts['transitions']:
                                counts['transitions'][f'{prev_tag}, {tag}'] = 1
                            else:
                                counts['transitions'][f'{prev_tag}, {tag}'] += 1

                        if word != '':
                            if f'{tag}, {word}' not in counts['emissions']:
                                counts['emissions'][f'{tag}, {word}'] = 1
                            else:
                                counts['emissions'][f'{tag}, {word}'] += 1

            total_counts[lang] = counts

        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(total_counts, output_file, ensure_ascii=False, indent=4)

    return total_counts


def calculate_emission_probs(tag_counts: dict, tag_word_counts: dict):
    """
    Calculate the emission probabilities and saves it on a file if the file doesn't exist
    :param tag_counts: json of the counts of the tags
    :param tag_word_counts: json of counts of a word given the tag
    :return: emission probabilites
    """

    emission_probs = {}
    # iterate over tag_word pairs to find the probability of each
    for key, counts in tag_word_counts.items():
        tag, word = key.split(', ')  # Ex. "NOUN, oro"
        count_of_tag = tag_counts[tag]
        prob = counts/count_of_tag  # counts(tag, word)/counts(tag)
        emission_probs[key] = prob

    with open("emission_mat.json", "w") as archivo:
        json.dump(emission_probs, archivo, ensure_ascii=False, indent=4)
    return emission_probs


def calculate_transition_probs(tag_counts: dict, tag_tag_counts: dict):
    """
    Calculates the transition probs and saves it on a file if the file doesn't exist
    :param tag_counts: json of counts of tags
    :param tag_tag_counts: json of counts of tagtag
    :return: transition probabilities for that tag
    """
    # creamos matriz de transiciones
    trans_mat = {}
    # recorrer todas las posibles tagtagprobs
    for tag1_tag2, counts in tag_tag_counts.items():
        # print(tag1_tag2, counts)
        prev_tag, current_tag = tag1_tag2.split(", ")
        prev_tag_count = tag_counts.get(f"{prev_tag}")

        prob = counts / prev_tag_count
        trans_mat[tag1_tag2] = prob

    with open("trans_mat.json", "w") as archivo:
        json.dump(trans_mat, archivo, ensure_ascii=False, indent=4)
    return trans_mat


def predict_tags(sentence, trans_mat, emiss_mat):
    """
    Predict the most probability tags for the sentence
    :param sentence: sentence to be predicted
    :param trans_mat: matrix of prob of transitions
    :param emiss_mat: matrix of prob of emission
    :return: transition probabilities for that tag
    """

    pass


def main():
    total_counts = count_occurrences(Path('UD-Data'))
    trans_mat = calculate_transition_probs(total_counts["English"]["tags"], total_counts["English"]["transitions"])
    emission_mat = calculate_emission_probs(total_counts["English"]["tags"], total_counts["English"]["emissions"])


if __name__ == "__main__":
    main()
