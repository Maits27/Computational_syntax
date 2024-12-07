import json, glob, os
from pathlib import Path

def establish_unk_words(lines, threshold=2):
    """
    This function establishes the words that will be considered as unknown
    :param lines: all the lines of the corpus
    :param threshold: threshold to consider a word as unknown
    :return: a list with the words that will be considered as unknown
    """
    words = {}
    unk_words = set()
    for line in lines:
        if line == '\n' or line[0] == '#':
            continue
        w_id, word, lemma, tag, _, _, _, tag2, _, _ = line.split('\t')
        if word not in words:
            words[word] = 1
        else:
            words[word] += 1
    for word, count in words.items():
        if count < threshold:
            unk_words.add(word)
    print(len(unk_words))
    return unk_words

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
        if not (os.path.exists('data')):
            os.mkdir('data')
        for lang in total_counts:
            counts = {'transitions': {}, 'emissions': {}, 'tags': {'<BOL>': 1}}
            with open(glob.glob(os.path.join(f'{corpus}/{lang}/', '*train.conllu'))[0], 'r', encoding='utf-8') as f:
                unk_words = establish_unk_words(f.readlines())
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
                        if word in unk_words:
                            word = '<UNK>'

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


def calculate_emission_probs(tag_counts: dict, tag_word_counts: dict, lang: str):
    """
    Calculate the emission probabilities and saves it on a file if the file doesn't exist
    :param tag_counts: json of the counts of the tags
    :param tag_word_counts: json of counts of a word given the tag
    :param lang: language
    :return: emission probabilites
    """

    emission_probs = {}
    # iterate over tag_word pairs to find the probability of each
    for key, counts in tag_word_counts.items():
        tag, word = key.split(', ')  # Ex. "NOUN, oro"
        count_of_tag = tag_counts[tag]
        prob = counts/count_of_tag  # counts(tag, word)/counts(tag)
        emission_probs[key] = prob

    with open(f"data/{lang}_emission_mat.json", "w") as archivo:
        json.dump(emission_probs, archivo, ensure_ascii=False, indent=4)
    return emission_probs


def calculate_transition_probs(tag_counts: dict, tag_tag_counts: dict, lang: str):
    """
    Calculates the transition probs and saves it on a file if the file doesn't exist
    :param tag_counts: json of counts of tags
    :param tag_tag_counts: json of counts of tagtag
    :param lang: language
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

    with open(f"data/{lang}_trans_mat.json", "w") as archivo:
        json.dump(trans_mat, archivo, ensure_ascii=False, indent=4)
    return trans_mat


def evaluate_model(input_path, trans_mat, emiss_mat, all_tags, lang='English'):
    """
    Evaluate the model with the test data
    :param input_path: path to the test data
    :param lang: language of the data
    :return: accuracy of the model
    """
    if not Path('output').exists():
        os.mkdir('output')
    with open(glob.glob(os.path.join(f'{input_path}/{lang}/', '*dev.conllu'))[0], 'r', encoding='utf-8') as f:
        predictions = []
        sentence, tags = '', ['<BOL>']
        for line in f:
            if line == '\n' or line[0] == '#':
                if sentence != '':
                    tags.append('<EOL>')
                    res = predict_tags(sentence[:-1], trans_mat, emiss_mat, all_tags)
                    predictions.append({'sentence': sentence[:-1], 'tags': tags, 'prediction': res})
                    sentence, tags = '', ['<BOL>']
            else:
                w_id, word, lemma, tag, _, _, _, tag2, _, _ = line.split('\t')
                sentence += f'{word} '
                tags.append(tag)
    with open(f'output/{lang}_predictions.jsonl', 'w', encoding='utf-8') as output_file:
        for p in predictions:
            output_file.write(json.dumps(p, ensure_ascii=False, indent=4) + '\n')


def predict_tags(sentence, trans_mat, emiss_mat, tags, lang='English'):
    """
    Predict the most probability tags for the sentence
    :param sentence: sentence to be predicted
    :param trans_mat: matrix of prob of transitions
    :param emiss_mat: matrix of prob of emission
    :return: transition probabilities for that tag
    """

    words = sentence.replace('.', '').lower().split(' ')
    result = ['<BOL>']
    for w in words:
        max_prob = 0
        max_prob_tag = ''

        # iterate all the posible tags
        for tag in tags:
            if f'{tag}, {w}' in emiss_mat: #get prob. of emission matrix (TAG, word)
                prob = emiss_mat[f'{tag}, {w}']
                if f'{result[-1]}, {tag}' in trans_mat: # get prob of transition matrix (TAG1, TAG2)
                    prob = prob * trans_mat[f'{result[-1]}, {tag}'] # mulitply both probs
                    if prob > max_prob: #if it is bigger than the max, apply
                        max_prob = prob
                        max_prob_tag = tag


        #check if it is UNK word
        if max_prob_tag == '':
            for tag in tags:
                if f'{tag}, <UNK>' in emiss_mat: # check with the <UNK> token
                    prob = emiss_mat[f'{tag}, <UNK>']
                    if f'{result[-1]}, {tag}' in trans_mat:
                        prob = prob * trans_mat[f'{result[-1]}, {tag}']
                        if prob > max_prob:
                            max_prob = prob
                            max_prob_tag = tag

        result.append(max_prob_tag)
    result.append('<EOL>')
    return result


def predict_examples():
    #predict a example of each language via terminal
    for lang in ['English',"Spanish"]:
        with open(f"data/{lang}_emission_mat.json", "r", encoding="utf-8") as archivo:
            em = json.loads(archivo.read())
        with open(f"data/{lang}_trans_mat.json", "r") as archivo:
            tm = json.loads(archivo.read())
        with open("data/counts.json", "r") as archivo:
            counts = json.loads(archivo.read())
        sentence = input(f"Write a sentence in {lang} or press ENTER to default one: ")
        if sentence == '':
            if lang == 'English':
                sentence = "i would like the cheapest flight from pittsburgh to atlanta leaving april twenty fifth and returning may sixth"
            else:
                sentence = "La intérprete de No me importa nada llega mañana , a las diez de la noche , al a el Palau ."
        res = predict_tags(sentence, tm, em, counts[lang]["tags"].keys())
        words = sentence.split()  # Separate in words the sentence
        tags = res[1:-1]
        for word, tag in zip(words, tags):
            print(f"{word:<15}{tag}")


def main():

    # CREATE COUNTS FOR EACH LANGUAGE
    total_counts = count_occurrences(Path('UD-Data'))

    # TRAIN and Evaluate each language
    for lang in ["English", "Spanish"]:
        trans_mat = calculate_transition_probs(total_counts[lang]["tags"], total_counts[lang]["transitions"], lang)
        emission_mat = calculate_emission_probs(total_counts[lang]["tags"], total_counts[lang]["emissions"], lang)

        evaluate_model(Path('UD-Data'), trans_mat, emission_mat, total_counts[lang]["tags"].keys(), lang)


if __name__ == "__main__":
    #option to train or predict examples
    option = input("Train (t) or Predict (p): ")
    if option == 't':
        main()
    elif option == 'p':
        predict_examples()
    else:
        print("Invalid option")

