from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import pickle

# Define the special tokens
special_tokens_list = ["<START>", "<END>", "<PAD>", "<UNK>", "<SEP>", "<CLS>", "<MASK>"]

# Initialize the BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

# Set the pre-tokenizer to split the inputs into words
tokenizer.pre_tokenizer = Whitespace()

# Register the special tokens to the tokenizer
tokenizer.add_special_tokens(special_tokens_list)

# Initialize the trainer with the special tokens
trainer = BpeTrainer(special_tokens=special_tokens_list)

# Load your dataset
dataset = pickle.load(open("./datasets/eng-ge/english-german-train.pkl", "rb"))

# Create lists of English and German sentences
eng = [i[0] for i in dataset]
germ = [i[1] for i in dataset]

# Combine the English and German sentences
both = eng + germ

# Train the tokenizer
tokenizer.train_from_iterator(both, trainer=trainer)

# Save the tokenizer to a file
tokenizer.save("./tokenizer-eng-de.json", pretty=True)
