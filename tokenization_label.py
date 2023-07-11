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
    tokenized_inputs = tokenizer(sen_entities_test["text"],  padding=True, truncation=True, max_length=512, is_split_into_words=True)
    labels = []
    for i, label in enumerate(sen_entities_test["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs
