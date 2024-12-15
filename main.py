import json, glob, os, re
from pathlib import Path

from typing import List, Dict


def split_sentence(sentence):
    """
    Split a sentence into words
    :param sentence: sentence to be splitted
    :return: list of words
    """
    #sentence = re.sub(r'(?<!\d)(?<!\w)([.,!?;:])(?!\d)(?!\w)', r' \1 ', sentence)
    #sentence = re.sub(r'\s+', ' ', sentence).strip()
    #sentence = sentence.replace('. . .', '...')
    words = sentence.lower().split(' ')
    return words


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
        if not re.fullmatch(r"-?\d+", w_id):
            #print(f'UNK_Ignoring line: {line}')
            continue
        if word not in words:
            words[word] = 1
        else:
            words[word] += 1
    for word, count in words.items():
        if count < threshold:
            unk_words.add(word)
    print(len(unk_words))
    return unk_words


def count_occurrences(corpus, steps: List[str], write=True):
    """
    This function counts the occurrences of each
        tag,
        word, tag
        tag, tag
    and saves it on a file if it doens't exist (if file already exists, it is not saved)
    :param corpus: the path where the corpus is located
    :param steps: the documents to be used to train the model
    :param write: if the file should be saved (or checked if it already exists)
    :return: returns a json with the counts ({tags:{...}, ...})
    """
    total_counts = {'English': {}, 'Spanish': {}}
    output_path = Path('data/counts.json')

    if output_path.exists() and write:
        with open(output_path, 'r', encoding='utf-8') as f:
            total_counts = json.load(f)
    else:
        if not (os.path.exists('data')):
            os.mkdir('data')
        for lang in total_counts:
            files_to_train = []
            for step in steps:
                files_to_train += glob.glob(os.path.join(f'{corpus}/{lang}/', f'*{step}.conllu'))
            counts = {'transitions': {}, 'emissions': {}, 'tags': {'<BOL>': 1}}

            unk_words = set()
            for file in files_to_train:
                with open(file, 'r', encoding='utf-8') as f:
                    unk_words = unk_words.union(establish_unk_words(f.readlines()))

            for file in files_to_train:
                with open(file, 'r', encoding='utf-8') as f:
                    tag = '<BOL>'
                    for line in f:
                        ignore_line = False
                        prev_tag, word = tag, ''
                        if line == '\n':
                            tag = '<EOL>'
                        elif line[0] == '#':
                            tag = '<BOL>'
                        else:
                            wordL = line.split('\t')
                            if not re.fullmatch(r"-?\d+", wordL[0]):

                                #print(f'COUNT_Ignoring line: {line}')
                                ignore_line = True
                                continue
                            else:
                                w_id, word, lemma, tag, _, _, _, tag2, _, _ = line.split('\t')
                                word = word.lower()
                                if word in unk_words:
                                    word = '<UNK>'

                        if not ignore_line:
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
        if write:
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
        prob = counts / count_of_tag  # counts(tag, word)/counts(tag) # np.log2(counts)-np.log2(count_of_tag)
        emission_probs[key] = prob  # np.log2(prob)

    with open(f"data/{lang}_emission_mat.json", "w", encoding="utf-8") as archivo:
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

    with open(f"data/{lang}_trans_mat.json", "w", encoding="utf-8") as archivo:
        json.dump(trans_mat, archivo, ensure_ascii=False, indent=4)
    return trans_mat


def evaluate_model(input_path, trans_mat, emiss_mat, all_tags, lang='English', info="", step='dev'):
    """
    Evaluate the model with the test data
    :param input_path: input path of the test data
    :param trans_mat: transition matrix in JSON format
    :param emiss_mat: emission matrix in JSON format
    :param all_tags: possible tag list
    :param lang: language (English or Spanish)
    :param info: additional info to save the file
    :param step: step of the evaluation (dev or test)
    :return:
    """
    if not Path('output').exists():
        os.mkdir('output')
    if len(glob.glob(os.path.join(f'{input_path}/{lang}/', f'*{step}.conllu'))) == 0:
        p = os.path.join(f'{input_path}/{lang}/', f'*{step}.conllu')
        print(f'{p} file not found')
        return
    with open(glob.glob(os.path.join(f'{input_path}/{lang}/', f'*{step}.conllu'))[0], 'r', encoding='utf-8') as f:
        predictions = []
        sentence, tags = '', []
        for line in f:
            if line == '\n' or line[0] == '#':
                if sentence != '':

                    res = predict_tags(sentence[:-1], trans_mat, emiss_mat, all_tags)
                    predictions.append({'sentence': sentence[:-1], 'tags': tags, 'prediction': res})
                    sentence, tags = '', []
            else:
                w_id, word, lemma, tag, _, _, _, tag2, _, _ = line.split('\t')
                if not re.fullmatch(r"-?\d+", w_id):
                    #print(f'EVAl- ignoring line: {line}')
                    pass
                else:
                    sentence += f'{word} '
                    tags.append(tag)

    with open(f'output/{info}{lang}_predictions.jsonl', 'w', encoding='utf-8') as output_file:
        for p in predictions:
            output_file.write(json.dumps(p, ensure_ascii=False, indent=4) + '\n')


def predict_tags(sentence: str, trans_mat: Dict[str, float], emiss_mat: Dict[str, float], tags: List[str]):
    """
    Predict the most probability tags for the sentence
    :param sentence: sentence to be predicted
    :param trans_mat: matrix of prob of transitions
    :param emiss_mat: matrix of prob of emission
    :param tags: list of possible tags
    :return: POS tag sequence for the input sentence
    """
    sentence.lower()
    assert len(sentence.replace(" ", "")) > 0, "The sentence must contain at least one token"

    set_tags = set(tags)
    set_tags.remove("<EOL>")
    set_tags.remove("<BOL>")
    try_tags = list(set_tags)

    words = split_sentence(sentence.lower())
    # words.append('<EOL>')

    result = []
    probabilidades = [[(0.0, 'TAG_INVENTADO_PRUEBA', -1) for _ in range(len(try_tags))] for _ in range(len(words))]
    existe_palabra = False
    # the first word
    w = words[0]
    for j, tag_actual in enumerate(try_tags):
        p_acc = 1
        if f'{tag_actual}, {w}' in emiss_mat:
            existe_palabra = True
            p_e = emiss_mat[f'{tag_actual}, {w}']
            p_t = trans_mat[f'<BOL>, {tag_actual}'] if f'<BOL>, {tag_actual}' in trans_mat else 0
            p_actual = p_acc * p_e * p_t
            probabilidades[0][j] = (p_actual, tag_actual, -1)

    if not existe_palabra:
        for j, tag_actual in enumerate(try_tags):
            p_acc = 1
            if f'{tag_actual}, <UNK>' in emiss_mat:
                p_e = emiss_mat[f'{tag_actual}, <UNK>']
                p_t = trans_mat[f'<BOL>, {tag_actual}'] if f'<BOL>, {tag_actual}' in trans_mat else 0
                p_actual = p_acc * p_e * p_t
                probabilidades[0][j] = (p_actual, tag_actual, -1)

    for i, w in enumerate(words[1:], start=1):
        #print(f"word: {word}, id {i}")
        existe_palabra = False
        # iterate all the posible actual tags
        for j, tag_actual in enumerate(try_tags):
            # other words
            # we iterate the previous tags and we choose the one with the max prob
            if f'{tag_actual}, {w}' in emiss_mat:
                for t, tag_anterior in enumerate(try_tags):
                    p_acc, _, _ = probabilidades[i - 1][t]
                    p_e = emiss_mat[f'{tag_actual}, {w}']
                    p_t = trans_mat[
                        f'{tag_anterior}, {tag_actual}'] if f'{tag_anterior}, {tag_actual}' in trans_mat else 0
                    p_actual = p_acc * p_e * p_t
                    if p_actual > probabilidades[i][j][0]:  # we save the max prob
                        probabilidades[i][j] = (p_actual, tag_anterior, t)
                        existe_palabra = True
            # check if it is UNK word
        if not existe_palabra:
            for j, tag_actual in enumerate(try_tags):
                if f'{tag_actual}, <UNK>' in emiss_mat:
                    for t, tag_anterior in enumerate(try_tags):
                        p_acc, _, _ = probabilidades[i - 1][t]
                        p_e = emiss_mat[f'{tag_actual}, <UNK>']
                        p_t = trans_mat[
                            f'{tag_anterior}, {tag_actual}'] if f'{tag_anterior}, {tag_actual}' in trans_mat else 0
                        p_actual = p_acc * p_e * p_t
                        # print(i, j)
                        if p_actual > probabilidades[i][j][0]:  # we save the max prob
                            probabilidades[i][j] = (p_actual, tag_anterior, t)

    max_prob = 0
    max_prob_tag = ''
    id_anterior = 0
    for t, tag in enumerate(try_tags):
        p_acc, tag_anterior, id_tag = probabilidades[-1][t]
        p_t = trans_mat[f'{tag}, <EOL>'] if f'{tag}, <EOL>' in trans_mat else 0
        p_actual = p_acc * p_t
        if p_actual > max_prob:  # we save the max prob
            max_prob = p_actual
            max_prob_tag = tag
            id_anterior = t

    # backtrack
    pila = []
    pila.append(max_prob_tag)
    for i in range(len(words)-1, 0, -1):
        _, max_prob_tag_anterior, id_anterior = probabilidades[i][id_anterior]
        pila.append(max_prob_tag_anterior)

    #print(probabilidades)
    pila.reverse()
    return pila


def predict_examples():
    """
    Method to try de model in English and Spanish
    :return: Void
    """
    for lang in ['English', "Spanish"]:
        with open(f"data/{lang}_emission_mat.json", "r", encoding="utf-8") as archivo:
            em = json.loads(archivo.read())
        with open(f"data/{lang}_trans_mat.json", "r", encoding="utf-8") as archivo:
            tm = json.loads(archivo.read())
        with open("data/counts.json", "r", encoding="utf-8") as archivo:
            counts = json.loads(archivo.read())
        sentence = input(f"Write a sentence in {lang} or press ENTER to default one: ")
        if sentence == '':
            if lang == 'English':
                sentence = "i would like the cheapest flight from pittsburgh to atlanta, leaving april twenty fifth and returning may sixth. "
            else:
                sentence = "La intérprete de No me importa nada llega mañana, a las diez de la noche, al a el Palau."
        res = predict_tags(sentence, tm, em, counts[lang]["tags"].keys())

        words = split_sentence(sentence)  # Separate in words the sentence

        tags = res
        for word, tag in zip(words, tags):
            print(f"{word:<15}{tag}")


def main(steps: List[str]):
    # CREATE COUNTS FOR EACH LANGUAGE
    total_counts = count_occurrences(Path('UD-Data'), steps)

    # TRAIN and Evaluate each language
    for lang in ["English", "Spanish"]:
        trans_mat = calculate_transition_probs(total_counts[lang]["tags"], total_counts[lang]["transitions"], lang)
        emission_mat = calculate_emission_probs(total_counts[lang]["tags"], total_counts[lang]["emissions"], lang)

        step = 'test' if len(steps) == 2 else 'dev'
        evaluate_model(Path('UD-Data'), trans_mat, emission_mat, total_counts[lang]["tags"].keys(), lang, f'{step}_',
                       step)
        # out of domain evaluation
        evaluate_model(Path('UD-Data/out_of_domain'), trans_mat, emission_mat, total_counts[lang]["tags"].keys(), lang,
                       "od_")


if __name__ == "__main__":
    # option to train or predict examples
    option = input(
        "CHOOSE ONE:\n\nTrain and evaluate (e)\nTrain with dev too and run test (t)\nPredict (p)\nYour option: ")
    if option == 't':
        main(['train'])
    elif option == 'e':
        main(['train', 'dev'])
    elif option == 'p':
        predict_examples()
    else:
        print("Invalid option")
