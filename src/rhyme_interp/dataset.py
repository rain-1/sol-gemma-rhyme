"""A small controlled benchmark of natural couplet completions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .rhyme import rhymes


@dataclass(frozen=True)
class Example:
    id: str
    anchor: str
    target: str
    first_line: str
    final_prefix: str
    wrapper: str

    @property
    def prompt(self) -> str:
        return self.wrapper.format(first=self.first_line, second=self.final_prefix)

    def to_dict(self) -> dict[str, str]:
        return {**asdict(self), "prompt": self.prompt}


# Each line ends at the rhyme-bearing word. Targets are deliberately common words.
COUPLETS = [
    ("light", "night", "The window held a square of light", "An owl went hunting through the"),
    ("moon", "tune", "A silver cloud uncovered the moon", "The old musician played a"),
    ("sky", "high", "A hawk drew circles in the sky", "It rode the warming current"),
    ("rain", "train", "The roof began to drum with rain", "Beyond the fields there passed a"),
    ("breeze", "trees", "The curtains stirred beneath the breeze", "A whisper traveled through the"),
    ("flame", "name", "The candle woke a tiny flame", "Beside it someone carved a"),
    ("stream", "dream", "The willows leaned above the stream", "The child awoke from one last"),
    ("bell", "shell", "The harbor rang its evening bell", "A hermit crab withdrew inside its"),
    ("road", "load", "The wagon rolled along the road", "The patient oxen pulled their"),
    ("star", "guitar", "At dusk we saw the first bright star", "A traveler stopped to play his"),
    ("snow", "glow", "The silent garden filled with snow", "The cottage windows cast a"),
    ("sea", "free", "The river hurried toward the sea", "The captive bird at last flew"),
    ("day", "away", "The lark announced the newborn day", "The final shadows slipped"),
    ("stone", "alone", "A violet grew beside the stone", "It bloomed there quietly"),
    ("fire", "desire", "We gathered close around the fire", "The singer spoke of lost"),
    ("door", "more", "A stranger waited by the door", "He knocked and softly asked once"),
    ("blue", "true", "The lake reflected endless blue", "Her solemn promise still rang"),
    ("ground", "sound", "The acorn tumbled to the ground", "We listened for the smallest"),
    ("air", "care", "The scent of roses filled the air", "She trimmed each blossom patiently with"),
    ("face", "place", "A smile appeared upon his face", "He knew that he had found his"),
    ("eyes", "skies", "The dawn was mirrored in her eyes", "Pink clouds went sailing through the"),
    ("hill", "still", "The shepherd climbed the grassy hill", "At sunset every field grew"),
    ("time", "climb", "The old clock marked the passing time", "The patient ivy learned to"),
    ("cold", "old", "The northern wind blew sharp and cold", "The cabin stood there, dark and"),
    ("deep", "sleep", "The roots ran hidden, dark and deep", "The tired village settled down to"),
]

WRAPPERS = {
    "plain": "{first}\n{second}",
    "poem_label": "Poem:\n{first}\n{second}",
    "complete": "Complete the final word of this rhyming couplet.\n{first}\n{second}",
    "couplet": "Here is a rhyming couplet:\n{first}\n{second}",
    "literary": "Two lines of verse:\n{first}\n{second}",
    "quoted": "\"{first}\n{second}",
    "lower_instruction": "complete the rhyme:\n{first}\n{second}",
    "explicit": "Make the last word rhyme with the previous line.\n{first}\n{second}",
}

# Six completed lines used to elicit poem continuation from a base model. They
# form three adjacent rhyming couplets before the held-out target couplet.
RHYME_DEMONSTRATION_LINES = [
    "The rain made mirrors in the lane",
    "At dusk there came a distant train",
    "The fire faded to a glow",
    "Beyond the glass descended snow",
    "A gull went wheeling by the sea",
    "It spread its wings and wandered free",
]

# Claude Haiku generated these as independent two-line poems. They were then
# checked with CMUdict below in the test suite. Unlike the first prompt sweep,
# no couplet is recycled when the requested context grows.
DISTINCT_DEMONSTRATION_COUPLETS = [
    ("The children ran out to go and play", "They laughed and danced along their merry way"),
    ("She opened up her favorite dusty book", "He gave the painting one long careful look"),
    ("The engine roared to life with a sudden start", "Music filled the dancer's beating heart"),
    ("She knocked upon the old oak wooden door", "He wanted nothing but to help her more"),
    ("The baker sliced the warm and golden bread", "Come join us now she kindly said"),
    ("The campfire burned with orange flame so dire", "The smoke rose up and climbed much higher"),
    ("He carved his name deep in the ancient stone", "A house to call completely all his own"),
    ("The ocean stretched before us deep and blue", "A love that lasts forever tried and true"),
    ("The trees bent to the rushing winter wind", "He could not get the painful memory from his mind"),
    ("The sunset painted clouds in shades of gold", "The merchant's secret vault of stories old"),
    ("She hummed a sweet and gentle haunting song", "A bond between two hearts both brave and strong"),
    ("The roots of ancient trees dig far and deep", "The promises we make are ours to keep"),
    ("The orange cat sat on my mat", "A mouse took cover where we sat"),
    ("Birds nested high in the tall tree", "The king proclaimed an ancient decree"),
    ("Children ran fast beneath the sun", "They laughed and played till they were done"),
    ("The farmer lived with his dear spouse", "Their cat would chase the sneaky mouse"),
    ("The acrobat performed a great jump", "He crashed down with an awful thump"),
    ("She wore a dress to the dance", "She hoped to find a new romance"),
    ("The boxer stepped into the ring", "His corner coach encouraged him to swing"),
    ("The water moved in a gentle flow", "The green plants started then to grow"),
    ("The delicate flowers arrived in spring", "The butterfly stretched out its painted wing"),
    ("They dove into the sparkling lake", "The surface ripples that they would make"),
    ("The climber scaled the jagged rock", "And heard the ticking of the clock"),
    ("The woman put on a nice dress", "The kids created such a mess"),
]


def _last_word(line: str) -> str:
    return line.rsplit(maxsplit=1)[-1].lower()


def distinct_demonstrations(anchor: str, count: int) -> list[tuple[str, str]]:
    """Choose distinct demos without the target anchor's rhyme family.

    Excluding same-family demonstrations prevents a longer prompt from directly
    priming candidate answers for that target.
    """
    eligible = [
        pair
        for pair in DISTINCT_DEMONSTRATION_COUPLETS
        if not rhymes(anchor, _last_word(pair[0])) and not rhymes(anchor, _last_word(pair[1]))
    ]
    if count > len(eligible):
        raise ValueError(f"Only {len(eligible)} non-overlapping demonstrations are available for {anchor}")
    return eligible[:count]


def build_distinct_elicitation_dataset(count: int, shuffled: bool = False) -> list[Example]:
    """Build target prompts preceded by `count` unique demonstration poems."""
    examples = []
    condition = f"distinct_{count}" + ("_shuffled" if shuffled else "")
    for i, (anchor, target, first, second) in enumerate(COUPLETS):
        pairs = distinct_demonstrations(anchor, count)
        if shuffled and len(pairs) > 1:
            second_lines = [pair[1] for pair in pairs]
            second_lines = second_lines[1:] + second_lines[:1]
            lines = [line for pair, second_line in zip(pairs, second_lines) for line in (pair[0], second_line)]
        else:
            lines = [line for pair in pairs for line in pair]
        prefix = "\n".join(lines) + "\n"
        examples.append(Example(f"{i:02d}-{condition}", anchor, target, first, second, prefix + "{first}\n{second}"))
    return examples


def elicitation_context(condition: str) -> str:
    """Return a controlled prefix for the longer-context experiment."""
    lines = RHYME_DEMONSTRATION_LINES
    contexts = {
        "plain": "",
        "rhyming": "\n".join(lines) + "\n",
        # Same six lines and vocabulary, but the adjacent line endings no longer
        # rhyme. This tests rhyme-pattern induction separately from extra context.
        "shuffled": "\n".join([lines[0], lines[3], lines[2], lines[5], lines[4], lines[1]]) + "\n",
        "reversed": "\n".join([lines[1], lines[0], lines[3], lines[2], lines[5], lines[4]]) + "\n",
    }
    try:
        return contexts[condition]
    except KeyError as exc:
        raise ValueError(f"Unknown elicitation condition: {condition}") from exc


def build_elicitation_dataset(condition: str = "rhyming") -> list[Example]:
    prefix = elicitation_context(condition)
    return [
        Example(f"{i:02d}-{condition}", anchor, target, first, second, prefix + "{first}\n{second}")
        for i, (anchor, target, first, second) in enumerate(COUPLETS)
    ]


def build_dataset() -> list[Example]:
    return [
        Example(f"{i:02d}-{name}", anchor, target, first, second, wrapper)
        for i, (anchor, target, first, second) in enumerate(COUPLETS)
        for name, wrapper in WRAPPERS.items()
    ]


def build_counterfactual_dataset() -> list[Example]:
    """Exact-scaffold prompts for causal patches between rhyme anchors.

    Every varying anchor is a single Pythia token. Keeping the scaffold fixed
    avoids mixing a rhyme intervention with changes in syntax or token position.
    """
    wrapper = (
        "Choose one word to complete the rhyme.\n"
        "The first line ends with: {first}\n"
        "The second line ends with:{second}"
    )
    return [
        Example(f"cf-{i:02d}", anchor, target, anchor, "", wrapper)
        for i, (anchor, target, _first, _second) in enumerate(COUPLETS)
    ]


def write_jsonl(path: str | Path, examples: list[Example] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for example in examples or build_dataset():
            handle.write(json.dumps(example.to_dict()) + "\n")
