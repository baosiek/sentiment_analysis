import bert.tokenization as tk


def create_tokenizer(vocab_file, do_lower_case=False):
    return tk.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)