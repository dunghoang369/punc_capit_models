import os
import logging

import torch 
from torch.utils.data import TensorDataset

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_id_capit, label_id_punc, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id_capit = label_id_capit
        self.label_id_punc = label_id_punc
        self.valid_ids = valid_ids
        self.label_mask = label_mask


class NerProcessor:
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "valid.txt")), "valid")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return [["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"], ["B-COMMAS", "O", "B-DOT", "B-QUESTION"]]

    def _read_file(self, filename):
        '''
        read file
        '''
        f = open(filename)
        data = []
        sentence = []
        label = []

        for i, line in enumerate(f, 1):
            if not line.strip() or len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n" or line[0] == '.':
                if len(sentence) > 0:
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue

            splits = line.split()
            assert len(splits) >= 2, "error on line {}. Found {} splits".format(i, len(splits))
            word, tag = splits[0], splits[-1]
            # assert tag in self.get_labels(), "unknown tag {} in line {}".format(tag, i)
            sentence.append(word.strip())
            label.append(tag.strip())

        if len(sentence) > 0:
            data.append((sentence, label))
            sentence = []
            label = []
        return data

    def _create_examples(self, lines, set_type):
        examples = []

        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list_capit, label_list_punc, max_seq_length, encode_method):
    """Converts a set of examples into XLMR compatible format

    * Labels are only assigned to the positions correspoinding to the first BPE token of each word.
    * Other positions are labeled with 0 ("IGNORE")

    """
    ignored_label = "IGNORE"
    # label_map = {label: i for i, label in enumerate(label_list, 1)}
    # label_map[ignored_label] = 0  # 0 label is to be ignored
    label_map_capit = {label: i for i, label in enumerate(label_list_capit, 1)}
    label_map_capit[ignored_label] = 0
    label_map_punc = {label: i for i, label in enumerate(label_list_punc, 1)}
    label_map_punc[ignored_label] = 0
    
    features = []
    for (ex_index, example) in enumerate(examples):

        textlist = example.text_a.split(' ')
        labellist_capit, labellist_punc = [tags.strip().split("||")[0] for tags in example.label], [tags.strip().split("||")[1] for tags in example.label]
        labels_capit = []
        labels_punc = []
        valid = []
        label_mask = []
        token_ids = []
       
        for i, word in enumerate(textlist):  
            tokens = encode_method(word.strip())  # word token ids   
            token_ids.extend(tokens)  # all sentence token ids
            label_capit_1 = labellist_capit[i]
            label_punc_1 = labellist_punc[i]
            for m in range(len(tokens)):
                if m == 0:  # only label the first BPE token of each work
                    labels_capit.append(label_capit_1)
                    labels_punc.append(label_punc_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    labels_capit.append(ignored_label)  # unlabeled BPE token
                    labels_punc.append(ignored_label)
                    label_mask.append(0)
                    valid.append(0)

        logging.debug("token ids = ")
        logging.debug(token_ids)
        logging.debug("labels_capit = ")
        logging.debug(labels_capit)
        logging.debug("labels_punc = ")
        logging.debug(labels_punc)
        logging.debug("valid = ")
        logging.debug(valid)

        if len(token_ids) >= max_seq_length - 1:  # trim extra tokens
            token_ids = token_ids[0:(max_seq_length-2)]
            labels_capit = labels_capit[0:(max_seq_length-2)]
            labels_punc = labels_punc[0:(max_seq_length-2)]
            valid = valid[0:(max_seq_length-2)]
            label_mask = label_mask[0:(max_seq_length-2)]

        # adding <s>
        token_ids.insert(0, 0)
        labels_capit.insert(0, ignored_label)
        labels_punc.insert(0, ignored_label)
        label_mask.insert(0, 0)
        valid.insert(0, 0)

        # adding </s>
        token_ids.append(2)
        labels_capit.append(ignored_label)
        labels_punc.append(ignored_label)
        label_mask.append(0)
        valid.append(0)

        assert len(token_ids) == len(labels_capit)
        assert len(valid) == len(labels_capit)
        assert len(token_ids) == len(labels_punc)
        assert len(valid) == len(labels_punc)

        label_ids_capit = []
        for i, _ in enumerate(token_ids):
            label_ids_capit.append(label_map_capit[labels_capit[i]])

        label_ids_punc = []
        for i, _ in enumerate(token_ids):
            label_ids_punc.append(label_map_punc[labels_punc[i]])

        assert len(token_ids) == len(label_ids_capit)
        assert len(valid) == len(label_ids_capit)
        assert len(token_ids) == len(label_ids_punc)
        assert len(valid) == len(label_ids_punc)

        input_mask = [1] * len(token_ids)

        while len(token_ids) < max_seq_length:
            token_ids.append(1)  # token padding idx
            input_mask.append(0)
            label_ids_capit.append(label_map_capit[ignored_label])  # label ignore idx
            label_ids_punc.append(label_map_punc[ignored_label])
            valid.append(0)
            label_mask.append(0)

        while len(label_ids_capit) < max_seq_length:
            label_ids_capit.append(label_map_capit[ignored_label])
            label_mask.append(0)

        while len(label_ids_punc) < max_seq_length:
            label_ids_punc.append(label_map_punc[ignored_label])
            label_mask.append(0)

        assert len(token_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(label_ids_capit) == max_seq_length
        assert len(label_ids_punc) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        # if ex_index < 2:
        #     logging.info("*** Example ***")
        #     logging.info("guid: %s" % (example.guid))
        #     logging.info("tokens: %s" % " ".join(
        #         [str(x) for x in token_ids]))
        #     logging.info("input_ids: %s" %
        #                  " ".join([str(x) for x in token_ids]))
        #     logging.info("input_mask: %s" %
        #                  " ".join([str(x) for x in input_mask]))
        #     logging.info("label_capit: %s (id = %s)" % (example.label, " ".join(map(str, label_ids_capit))))
        #     logging.info("label_punc: %s (id = %s)" % (example.label, " ".join(map(str, label_ids_punc))))
        #     logging.info("label_mask: %s" %
        #                  " ".join([str(x) for x in label_mask]))
        #     logging.info("valid mask: %s" %
        #                 " ".join([str(x) for x in valid]))

        features.append(
            InputFeatures(input_ids=token_ids,
                          input_mask=input_mask,
                          label_id_capit=label_ids_capit,
                          label_id_punc=label_ids_punc,
                          valid_ids=valid,
                          label_mask=label_mask))

    return features


def create_dataset(features):
    
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_label_ids_capit = torch.tensor(
        [f.label_id_capit for f in features], dtype=torch.long)
    all_label_ids_punc = torch.tensor(
        [f.label_id_punc for f in features], dtype=torch.long)
    all_valid_ids = torch.tensor(
        [f.valid_ids for f in features], dtype=torch.long)
    all_lmask_ids = torch.tensor(
        [f.label_mask for f in features], dtype=torch.long)

    return TensorDataset(
        all_input_ids, all_label_ids_capit, all_label_ids_punc, all_lmask_ids, all_valid_ids)