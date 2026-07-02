from pathlib import Path
import random


def build_bigrams(text):
    bigrams = []

    for i in range(len(text) - 1):
        bigrams.append((text[i], text[i + 1]))

    return bigrams
# =========================
# Transition Inspector
# =========================
def build_transitions(bigrams):
    transitions = {}

    for current, next_char in bigrams:

        if current not in transitions:
            transitions[current] = {}

        if next_char not in transitions[current]:
            transitions[current][next_char] = 1
        else:
            transitions[current][next_char] += 1

    return transitions
# =========================
# Build Probabilities
# =========================
def build_probabilities(transitions):
    probabilities = {}

    for current, next_chars in transitions.items():

        total = sum(next_chars.values())

        probabilities[current] = {}

        for next_char, count in next_chars.items():
            probabilities[current][next_char] = count / total

    return probabilities
# =========================
# Top 10 Next Characters
# =========================
def print_top10(character, probabilities):

    if character not in probabilities:
        print(f"Character {repr(character)} tidak ditemukan.")
        return

    print(f"\nCharacter: {repr(character)}")

    top10 = sorted(
        probabilities[character].items(),
        key=lambda item: item[1],
        reverse=True
    )[:10]

    for next_char, prob in top10:
        print(f"{repr(next_char):>4} -> {prob:.4f}")
# =========================
# Text Generation
# =========================
def generate_text(probabilities, start_char="T", max_length=500):
    result = start_char
    current = start_char

    for _ in range(max_length):

        if current not in probabilities:
            break

        next_chars = list(probabilities[current].keys())
        weights = list(probabilities[current].values())

        current = random.choices(next_chars, weights=weights, k=1)[0]
        result += current

    return result

def main():

    text = Path("data/shakespeare.txt").read_text()

    bigrams = build_bigrams(text)

    print(f"Total bigrams : {len(bigrams)}")
    print(bigrams[:10])

    transitions = build_transitions(bigrams)

    print(transitions["a"])

    probabilities = build_probabilities(transitions)

    selected_chars = [
        "a",
        "e",
        "t",
        " ",
        "\n"
    ]

    for ch in selected_chars:
        print_top10(ch, probabilities)

    print("\n===== GENERATED TEXT =====\n")
    print(generate_text(probabilities))

if __name__ == "__main__":
    main()