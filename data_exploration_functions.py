import json

import numpy as np
from matplotlib import pyplot as plt
from wordcloud import WordCloud


def get_tags_predictions(ruta_archivo: str):
    """
    Get the tags and predictions of a dataset given a path of the predictions file
    :param ruta_archivo: path to a predictions file
    :return: all_tags: array with all tags
    :return: all_predictions: array with all predictions
    :return: errors: dictionary with the errors found
    """
    all_tags = []
    all_predictions = []

    # read de jsonl file and process it
    with open(ruta_archivo, "r", encoding="utf-8") as archivo:
        contenido = archivo.read()
        contenido = "[" + contenido.strip().replace("}\n{", "},{") + "]"
        elementos = json.loads(contenido)
        # print(len(elementos[0]))

    # get the tag and prediction of each instance and concatenate them
    errors = {}
    for elemento in elementos:
        correct_tags = elemento.get("tags", [])
        predicted_tags = elemento.get("prediction", [])
        for i in range(len(correct_tags)):
            correct_tag = correct_tags[i]
            predicted_tag = predicted_tags[i]
            if correct_tag != predicted_tag:
                if predicted_tag in ["ADJ", "NOUN", "VERB"]:
                    # we check what is the wrong tagged word
                    word = elemento.get("sentence", "").split()[i]
                    if f"{correct_tag},{predicted_tag}" not in errors:
                        errors[f"{correct_tag},{predicted_tag}"] = [word]
                    else:
                        errors[f"{correct_tag},{predicted_tag}"].append(word)

        all_tags.extend(elemento.get("tags", []))
        all_predictions.extend(elemento.get("prediction", []))
    return all_tags, all_predictions, errors


def conllu_dict(file_path, i=0):
    """
    Read a conllu file and return a dictionary with the tags and sentences
    :param file_path:
    :param i: default 0
    :return: dictionary with the tags and sentences
    """
    with open(file_path, "r", encoding="utf-8") as f:
        tags = []
        sentence = []
        info = {}
        for line in f:
            if line == '\n' and len(tags)!=0:
                info[i] = {"tags": tags, "sentence": sentence}
                i+=1
                tags, sentence = [], []
            elif line[0] == '#': pass
            else:
                w_id, word, lemma, tag, _, _, _, tag2, _, _ = line.split('\t')
                sentence.append(word.lower())
                tags.append(tag)
    return info, i


def counts_for_files(files, sentences, counting='Sentences'):
    """
    Plot the number of sentences per group
    :param files:
    :param sentences:
    :param counting:
    :return:
    """
    plt.figure(figsize=(5, 3))
    plt.bar(files, sentences, color='skyblue')
    plt.ylabel(f'Number of {counting}')
    plt.title(f'{counting} per group')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def sentence_length_boxplot(docs, lengths):
    """
    Plot a boxplot of the sentence lengths
    :param docs:
    :param lengths:
    :return: void
    """
    fig, ax = plt.subplots()
    box = ax.boxplot(lengths, vert=False, patch_artist=True)
    colors = ['lightcoral', 'skyblue', 'lightgreen', 'plum']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_yticklabels(docs)
    ax.set_xlabel('Sentence length')
    ax.set_title('Sentence length boxplot')
    plt.show()


def get_transition_matrix_and_labels(transitions):
    """
    Get the transition matrix and labels
    :param transitions:  dictionary with the transitions
    :return: transition matrix and labels
    """
    labels = set()
    for transition in transitions.keys():
        before, after = transition.split(", ")
        labels.add(before)
        labels.add(after)

    # Crear un índice para las etiquetas
    ordered_labels = ["<BOL>"] + [label for label in labels if label != "<BOL>" and label != "<EOL>"] + ["<EOL>"]
    labels = ordered_labels
    label_index = {label: i for i, label in enumerate(labels)}

    # Crear la matriz de transiciones (inicialmente llena de ceros)
    n = len(labels)
    matrix = np.zeros((n, n))

    # Rellenar la matriz con los datos de transiciones
    for transition, count in transitions.items():
        before, after = transition.split(", ")
        i = label_index[before]
        j = label_index[after]
        matrix[i, j] = count
    return matrix, labels


def transition_matrix(transitions):
    """
    Plot the transition matrix
    :param transitions:  dictionary with the transitions
    :return:  void
    """
    matrix, labels = get_transition_matrix_and_labels(transitions)

    plt.figure(figsize=(8, 6))
    cax = plt.imshow(matrix, cmap="viridis", interpolation="nearest")
    plt.colorbar(cax, label="Count")
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    plt.title("Transition matrix")
    plt.xlabel("Destination tag")
    plt.ylabel("Origin tag")


def plot_subsets_comparison(data, title):
    """
    Genera un gráfico de columnas para comparar train, dev y test por cada etiqueta (tag) en un idioma.
    :param title: Título del gráfico.
    :param data: Diccionario con los subconjuntos train, dev y test, y sus conteos por tag.
    formato: {'train' : counts_trains['English']['tags'],
        'dev' : counts_dev['English']['tags'],
        'test' : counts_test['English']['tags']}
    """
    subsets = list(data.keys())  # Detectar qué subconjuntos están presentes (e.g., train, dev, test)
    if not subsets:
        raise ValueError("El diccionario de datos está vacío. Proporcione al menos un subconjunto.")

    # Usar los tags del subconjunto principal como referencia y ordenarlos de mayor a menor
    reference_subset = 'train' if 'train' in data else subsets[0]
    tags = sorted(data[reference_subset].keys(), key=lambda tag: data[reference_subset][tag], reverse=True)

    if '<BOL>' in tags: tags.remove('<BOL>')
    if '<EOL>' in tags: tags.remove('<EOL>')

    counts = {subset: [data[subset].get(tag, 0) for tag in tags] for subset in subsets}

    x = np.arange(len(tags))  # Posiciones en el eje x
    width = 0.8 / len(subsets)  # Ancho dinámico de las barras según el número de subconjuntos

    plt.figure(figsize=(15, 6))
    colors = ['skyblue', 'orange', 'green']  # Colores para los subconjuntos

    for i, subset in enumerate(subsets):
        plt.bar(x + (i - len(subsets) / 2) * width, counts[subset], width, label=subset.capitalize(), color=colors[i % len(colors)])

    plt.yscale('log')
    plt.xlabel('Tags')
    plt.ylabel('Count')
    plt.xticks(x, tags, rotation=45, ha='right')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()


def wordclouds_by_tag(info):
    """
    Plot wordclouds by tag
    :param info:  dictionary with the tags and words
    :return:
    """
    num_por_linea = 4
    plt.figure(figsize=(15, 10), facecolor='none')

    for i, (categoria, palabras) in enumerate(info.items()):
        text = ' '.join(palabras)
        wordcloud = WordCloud(width=400, height=400, background_color=None, regexp=r'\S+').generate(text)

        plt.subplot(len(info) // num_por_linea + 1, num_por_linea, i + 1)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(categoria)
    plt.tight_layout()
    plt.show()


def wordcloud_of_words_or_sentences(sentences, title="Sentence wordcloud"):
    """
    Plot a wordcloud of words or sentences
    :param sentences:  array of strings or array of arrays with words
    :param title:  title of the plot
    :return:
    """
    joined_sentences = ' '
    if all(isinstance(words, list) for words in
           sentences):  # If it is an array of arrays with words [['I', 'am', 'a', 'sentence'], ['I', 'am', 'another', 'sentence']]
        sent_flat = [palabra for sublista in sentences for palabra in sublista]
        joined_sentences = ' '.join(sent_flat)
    elif all(isinstance(sentence, str) for sentence in
             sentences):  # If it is an array of strings ['I am a sentence', 'I am another sentence']
        joined_sentences = ' '.join(sentences)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(joined_sentences)
    plt.figure()
    plt.title(title)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def plot_pos_with_unknown(data,title,lang='English'):
    """
    Genera un gráfico de barras horizontales con la frecuencia de POS tags que contienen "<UNK>".
    :param data:  Diccionario con los counts.
    :param title: título del gráfico.
    :param lang: idioma del que se quiere visualizar los datos.
    :return:
    """
    emissions = data.get(lang, {}).get('emissions', {})

    # Filtramos las etiquetas POS que contienen "<UNK>"
    unk_tags = {tag: count for tag, count in emissions.items() if "<UNK>" in tag}

    # Ordenamos los resultados
    sorted_unk_tags = dict(sorted(unk_tags.items(), key=lambda item: item[1], reverse=True))

    tags = list(sorted_unk_tags.keys())
    counts = list(sorted_unk_tags.values())

    plt.figure(figsize=(10, 6))
    plt.barh(tags, counts, color='skyblue')
    plt.xlabel('Frecuencia')
    plt.title(title)

    plt.gca().invert_yaxis()
    plt.show()


