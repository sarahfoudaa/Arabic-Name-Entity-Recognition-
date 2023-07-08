from transformers import BertTokenizerFast

def tokenize_and_align_labels(sen_entities_test,tokenizer, label_all_tokens=True):
    """
    Function that tokenize the words in the dataframe and alight the labels with the splited words after tokenization,
    where each word after splitting have the same label as the original one beforw splitting

    Parameters:
      sen_entities_test():
      tokenizer():

    Returns:
      tokenized_inputs():
    """
    #https://huggingface.co/docs/transformers/pad_truncation
    tokenized_inputs = tokenizer(sen_entities_test["text"],  padding=True, truncation=True, max_length=512, is_split_into_words=True)
    labels = []
    for i, label in enumerate(sen_entities_test["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        # word_ids() => Return a list mapping the tokens to their actual word in the initial sentence.
        # It Returns a list indicating the word corresponding to each token.
        previous_word_idx = None
        label_ids = []
        # Special tokens like `<s>` and `<\s>` are originally mapped to None
        # We need to set the label to -100 so they are automatically ignored in the loss function.
        for word_idx in word_ids:
            if word_idx is None:
                # set –100 as the label for these special tokens
                label_ids.append(-100)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on the label_all_tokens flag.
            elif word_idx != previous_word_idx:
                # if current word_idx is != prev then its the most regular case and add the corresponding token
                label_ids.append(label[word_idx])
            else:
                # to take care of sub-words which have the same word_idx
                # set -100 as well for them, but only if label_all_tokens == False
                label_ids.append(label[word_idx] if label_all_tokens else -100)
                # mask the subword representations after the first subword
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs