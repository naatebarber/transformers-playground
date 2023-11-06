import torch
import torch.optim as optim
import torch.nn as nn
from tokenizers import Tokenizer
from tfs.transformer import Transformer
import pickle
from sklearn.utils import shuffle


dev = "cpu"

if torch.cuda.is_available():
    dev = "cuda:0"

if torch.backends.mps.is_available():
    dev = torch.device("mps")

device = torch.device(dev)

print(device)


def pseudo_train():
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1

    transformer = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
        device=device,
    )

    # Generate random sample data
    src_data = torch.randint(1, src_vocab_size, (64, max_seq_length)).to(
        device
    )  # (batch_size, seq_length)
    tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length)).to(
        device
    )  # (batch_size, seq_length)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )

    transformer.train()

    for epoch in range(100):
        optimizer.zero_grad()
        output = transformer.forward(src_data, tgt_data[:, :-1])
        loss = criterion(
            output.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        )
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")


def batch_iterator(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def translate_eng_de():
    tokenizer: Tokenizer = Tokenizer.from_file("./tokenizer-eng-de.json")

    src_vocab_size = tokenizer.get_vocab_size()
    tgt_vocab_size = src_vocab_size

    train_dataset = pickle.load(
        open("./datasets/eng-ge/english-german-train.pkl", "rb")
    )

    eng = [i[0] for i in train_dataset]
    deu = [i[1] for i in train_dataset]

    eng_tokens = tokenizer.encode_batch(eng)
    deu_tokens = tokenizer.encode_batch(deu)

    max_seq_length = max(
        max(len(output.ids) for output in eng_tokens),
        max(len(output.ids) for output in deu_tokens),
    )

    eng_tokens_padded = [
        output.ids
        + [tokenizer.token_to_id("<PAD>")] * (max_seq_length - len(output.ids))
        for output in eng_tokens
    ]
    deu_tokens_padded = [
        output.ids
        + [tokenizer.token_to_id("<PAD>")] * (max_seq_length - len(output.ids))
        for output in deu_tokens
    ]

    batch_size = 10

    eng_batches = [batch for batch in batch_iterator(eng_tokens_padded, 10)]
    deu_batches = [batch for batch in batch_iterator(deu_tokens_padded, 10)]

    d_model = 512
    num_heads = 8
    num_layers = 12
    d_ff = 2048
    max_seq_length = max_seq_length
    dropout = 0.1

    transformer = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(
        transformer.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9
    )

    transformer.train()

    for epoch in range(100):
        for batch in range(len(eng_batches)):
            eng_batch = eng_batches[batch]
            deu_batch = deu_batches[batch]

            eng_batch = torch.tensor(eng_batch).to(device)
            deu_batch = torch.tensor(deu_batch).to(device)

            optimizer.zero_grad()

            output = transformer.forward(eng_batch, deu_batch[:, :-1])
            loss = criterion(
                output.contiguous().view(-1, tgt_vocab_size),
                deu_batch[:, 1:].contiguous().view(-1),
            )
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")


translate_eng_de()
