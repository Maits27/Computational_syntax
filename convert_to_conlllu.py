import sys
import os


def convert_to_conll(input_file, output_file):
    """
    Convert dataset to CoNLL format with placeholders for empty columns and sentence boundaries.

    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to save the converted CoNLL file.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        sentence_counter = 0
        outfile.write(f"# sent_id = {sentence_counter}\n")
        word_counter = 1
        tokens = set()
        for line in infile:
            line = line.strip()

            # Sentence boundary: Insert comment with sentence ID
            if not line:
                sentence_counter += 1
                word_counter = 1
                outfile.write(f"\n# sent_id = {sentence_counter}\n")
                continue

            # Check if line contains valid token and POS tag
            parts = line.split()

            if len(parts) == 2:
                token, pos = parts
                # substitute wrong POS tags
                if pos in ["U", "#", "@", "~"]:
                    pos = "X"
                elif pos == "DETERMINER":
                    pos = "DET"
                elif pos == "INTERJECTION":
                    pos = "INTJ"
                elif pos == "E":
                    pos = "SYM"
                elif pos == ".":
                    pos = "PUNCT"
                elif pos == "CONJ":
                    if token.lower() in ["y", "o", "pero", "más", "mas", "ni", "sino", "e", "u", 'y', '-y', 'ni', 'pero', 'peeeeeero', 'sin embargo', 'and', 'peroo', 'nii', 'por', 'maas', 'incluso', 'peo', 'sii', 'peero', 'sea', 'pues', 'de', 'no', 'bien', 'pq', 'con', 'cuándo', 'ó', 'igual', 'pues', 'por', 'para', 'paraa', 'and', 'puuueeeeeees', 'queeee', 'pa', 'sisi', 'comoo', 'pa', 'ahora', 'comooooooooooooo', 'comoooooooooooo', 'a', 'holaaa', '¿porque', 'paso', 'peroo', 'quien', 'nomás', 'sii', 'ue', 'siii', 'quien', 'komo']:
                        pos = "CCONJ"
                    elif token.lower() in ["como", "q", "que", "mientras", "si", "sí","cuando", "ya que", "ya", "porque", 'aunque', 'porqe', 'porque', 'porqué', 'aunqe', 'quee', 'qu', 'qe', 'qie', 'qué',
    'porquee', 'aunque', 'aunque', 'aunque', 'pesé', 'pese', 'donde', 'cuando', 'cómo', 'quién', 'quienes', 'peroO', 'peró', 'peeeeeeeeeeero', 'ke', 'porq', 'a', 'k', 'peeero', 'aunq', 'qqque', 'aunq', 'esque', 'cual', 'porque', 'pesar', 'apesar', "salvo"]:
                        pos = "SCONJ"
                    else:
                        tokens.add(token.lower())

                # Write in CoNLL format with placeholders for empty columns
                outfile.write(f"{word_counter}\t{token}\t_\t{pos}\t_\t_\t_\t_\t_\t_\n")
                word_counter += 1
        print(tokens)
        # Ensure file ends with a newline
        outfile.write('\n')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_to_conll.py <input_file>")
        sys.exit(1)

    input_path = sys.argv[1]

    if not os.path.isfile(input_path):
        print(f"Error: The file '{input_path}' does not exist.")
        sys.exit(1)

    output_path = os.path.join(os.getcwd(), "UD-Data/out_of_domain/Spanish/es-tweets-dev.conllu")
    convert_to_conll(input_path, output_path)
    print(f"Converted file saved to: {output_path}")