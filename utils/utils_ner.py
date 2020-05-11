import logging
import os
import torch
from utils.utils_metrics import get_entities

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, valid_mask, segment_ids, label_ids, start_ids, end_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.valid_mask = valid_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.start_ids = start_ids
        self.end_ids = end_ids


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
    return examples


def convert_examples_to_features(
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    span_labels = []
    for label in label_list:
        label = label.split('-')[-1]
        if label not in span_labels:
            span_labels.append(label)
    span_map = {label: i for i, label in enumerate(span_labels)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        valid_mask = []
        for word in example.words:
            word_tokens = tokenizer.tokenize(word)
            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            for i, word_token in enumerate(word_tokens):
                if i == 0:
                    valid_mask.append(1)
                else:
                    valid_mask.append(0)
                tokens.append(word_token)
        label_ids = [label_map[label] for label in example.labels]
        entities = get_entities(example.labels)
        start_ids = [span_map['O']] * len(label_ids)
        end_ids = [span_map['O']] * len(label_ids)
        for entity in entities:
            start_ids[entity[1]] = span_map[entity[0]]
            end_ids[entity[-1]] = span_map[entity[0]]
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            valid_mask = valid_mask[: (max_seq_length - special_tokens_count)]
            start_ids = start_ids[: (max_seq_length - special_tokens_count)]
            end_ids = end_ids[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        start_ids += [pad_token_label_id]
        end_ids += [pad_token_label_id]
        valid_mask.append(1)
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            start_ids += [pad_token_label_id]
            end_ids += [pad_token_label_id]
            valid_mask.append(1)
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            start_ids += [pad_token_label_id]
            end_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
            valid_mask.append(1)
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            start_ids = [pad_token_label_id] + start_ids
            end_ids = [pad_token_label_id] + end_ids
            segment_ids = [cls_token_segment_id] + segment_ids
            valid_mask.insert(0, 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            start_ids = ([pad_token_label_id] * padding_length) + start_ids
            end_ids = ([pad_token_label_id] * padding_length) + end_ids
            valid_mask = ([0] * padding_length) + valid_mask
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
            start_ids += [pad_token_label_id] * padding_length
            end_ids += [pad_token_label_id] * padding_length
            valid_mask += [0] * padding_length
        while (len(label_ids) < max_seq_length):
            label_ids.append(pad_token_label_id)
            start_ids.append(pad_token_label_id)
            end_ids.append(pad_token_label_id)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(start_ids) == max_seq_length
        assert len(end_ids) == max_seq_length
        assert len(valid_mask) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("valid_mask: %s", " ".join([str(x) for x in valid_mask]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logger.info("start_ids: %s", " ".join([str(x) for x in start_ids]))
            logger.info("end_ids: %s", " ".join([str(x) for x in end_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          valid_mask=valid_mask,
                          segment_ids=segment_ids,
                          label_ids=label_ids,
                          start_ids=start_ids,
                          end_ids=end_ids)
        )
    return features

def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    batch_tuple = tuple(map(torch.stack, zip(*batch)))
    batch_lens = torch.sum(batch_tuple[1], dim=-1, keepdim=False)
    max_len = batch_lens.max().item()
    results = ()
    for item in batch_tuple:
        if item.dim() >= 2:
            results += (item[:, :max_len],)
        else:
            results += (item,)
    return results


def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
