import dspy

idx_labels_mapping = {0: 'neutral',
                      1: 'stigmatizing',
                      2: 'privileging'}

labels_idx_mapping = {'neutral': 0,
                      'stigmatizing': 1,
                      'privileging': 2}


def _create_examples(df):
    examples = [
        dspy.Example(word=ex.pattern, chunk=ex.text_preproc, valence=ex.label_class).with_inputs('chunk', 'word') for ex
        in df.itertuples()]
    return examples


def _create_sub_examples(df):
    sub_examples = [dspy.Example(words=[ex.pattern for ex in df.itertuples()],
                                 chunks=[ex.text_preproc for ex in df.itertuples()],
                                 valences=[ex.label_class for ex in df.itertuples()])]
    sub_examples = [x.with_inputs('words', 'chunks') for x in sub_examples]
    return sub_examples

