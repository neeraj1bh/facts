import matplotlib.pyplot as plt
import numpy as np
import textwrap
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Initialize and load embeddings
load_dotenv()
embeddings = OpenAIEmbeddings()


# Function to calculate L2 norm
def calculate_l2(v1, v2):
    return np.linalg.norm(v1 - v2) ** 2


# Function to wrap labels for better readability
def wrap_labels(labels, width):
    return ["\n".join(textwrap.wrap(label, width)) for label in labels]


# Function to plot the data
def plot(data, words):
    # Increase figure size here
    fig, ax = plt.subplots(figsize=(12, 8))  # Width: 12 inches, Height: 8 inches
    ax.imshow(data, cmap="Blues")

    labels = wrap_labels(words, 30)
    ax.set_xticks(np.arange(len(words)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(words)))
    ax.set_yticklabels(labels)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

    # Adding text inside the squares for better visibility
    for i in range(len(words)):
        for j in range(len(words)):
            ax.text(j, i, round(data[i, j], 2), ha="center", va="center", color="black")

    fig.tight_layout()
    plt.show()


# Example usage
words = [
    "The happy child jumped bravely from rock to rock",
    "The child was not timid and had a good time jumping from rock to rock",
    "Although filled with great fear, the child jumped from rock to rock",
    "sad",
    "happy",
    "cry",
    "laugh",
    "smile",
    "joy",
    "cheerful",
]

# Embedding the words
embs = [np.array(embeddings.embed_query(word)) for word in words]

# Calculating distances
data = np.array([[calculate_l2(e1, e2) for e1 in embs] for e2 in embs])

# Plotting
plot(data, words)
