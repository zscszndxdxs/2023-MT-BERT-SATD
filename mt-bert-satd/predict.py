from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.metrics import recall_score, precision_score

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import pandas as pd

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import process
import tokenization
from modeling_multitask_predict import BertConfig, BertForSequenceClassification
from optimization import BERTAdam

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
torch.cuda.set_device(0)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, dataset_label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            dataset_label: 1:imdb 2:yelp p 3:yelp f 4:trec 5:yahoo 6:ag 7:dbpedia
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.dataset_label = dataset_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, dataset_label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.dataset_label_id = dataset_label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_predict_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class AllProcessor(DataProcessor):

    def get_predict_examples(self, data_dir, task):
        predict_data = pd.read_csv(
            os.path.join("unclassified_files/" +data_dir + '.csv'), header=None, sep=",").values
        predict_data = np.delete(predict_data, 0, 0)
        print("=====================")
        print(predict_data)

        return self._create_examples(predict_data, "predict", task)

    def get_labels(self):
        """See base class."""
        return [["0", "1"]]

    def _create_examples(self, predict_data, set_type, task):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(predict_data):
            # if i>147:break
            guid = "%s-%s" % (set_type, i)
            
            text = str(line[0])
            text = process.getdata(text)
            if i <5:
                print(text)
            
            text_a = tokenization.convert_to_unicode(text)

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=None, dataset_label=task))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map_1 = {}
    for (i, label) in enumerate(label_list[0]):
        label_map_1[label] = i

    features = []

    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = None

        if example.dataset_label == "1":
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                    dataset_label_id=int(example.dataset_label)))
        if example.dataset_label == "2":
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                    dataset_label_id=int(example.dataset_label)))
        if example.dataset_label == "3":
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                    dataset_label_id=int(example.dataset_label)))
        if example.dataset_label == "4":
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                    dataset_label_id=int(example.dataset_label)))

        if example.dataset_label == "5":
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                    dataset_label_id=int(example.dataset_label)))

    return features


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task",
                        default=None,
                        type=str,
                        required=True,
                        help="The task you want to predict, 1-5",
                        )

    parser.add_argument(
			        "--data_dir",
			        default=None,
			        type=str,
			        required=True,
			        help="The input data dir. Should contain the .csv files (or other data files) for the task.",
			    )

    parser.add_argument("--batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--output_dir",
                        default="predict_results",
                        type=str,
                        required=True,
                        help="output files.")
    args = parser.parse_args()
    processor = AllProcessor()
    label_list = processor.get_labels()
    device = torch.device("cuda")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    torch.cuda.manual_seed_all(42)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = tokenization.FullTokenizer(
        vocab_file="well_trained_model/vocab.txt", do_lower_case=True)
    bert_config = BertConfig.from_json_file("well_trained_model/config.json")

    model = BertForSequenceClassification(bert_config, len(label_list))

    checkpoint = torch.load('well_trained_model/pytorch_model.bin')
    model.bert.load_state_dict(checkpoint['bert'])
    model.classifier_1.load_state_dict(checkpoint['classifier_1'])
    model.classifier_2.load_state_dict(checkpoint['classifier_2'])
    model.classifier_3.load_state_dict(checkpoint['classifier_3'])
    model.classifier_4.load_state_dict(checkpoint['classifier_4'])

    model.to(device)

    predict_examples = processor.get_predict_examples(args.data_dir, args.task)
    predict_features = convert_examples_to_features(predict_examples, label_list, 128, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in predict_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in predict_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in predict_features], dtype=torch.long)
    all_label_ids = torch.tensor([2 for f in predict_features], dtype=torch.long)
    all_dataset_label_ids = torch.tensor([f.dataset_label_id for f in predict_features], dtype=torch.long)
    predict_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_dataset_label_ids)
    predict_dataloader = DataLoader(predict_data, batch_size=args.batch_size, shuffle=False)

    model.eval()
    predict_list = []
    
    with open(os.path.join(args.output_dir, "results_" + args.data_dir + str(args.task) + ".txt"), "w") as f:
        for input_ids, input_mask, segment_ids, label_ids, dataset_label_id in predict_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = None
            
           
            with torch.no_grad():

                if str(args.task) == '1' or str(args.task) == '2' or str(args.task) == '3' or str(args.task) == '4':
                    logits = model(input_ids, segment_ids, input_mask, label_ids, dataset_label_id, task=args.task)
                    logits = logits.detach().cpu().numpy()
                    outputs = np.argmax(logits, axis=1)
                    for output in outputs:
                        f.write(str(output) + "\n")
                    predict_list.extend(outputs)
                if str(args.task) == '5':
                    logits1, logits2, logits3, logits4 = model(input_ids, segment_ids, input_mask, label_ids,
                                                               dataset_label_id, task=args.task)
                    logits1 = logits1.detach().cpu().numpy()
                    outputs1 = np.argmax(logits1, axis=1)

                    logits2 = logits2.detach().cpu().numpy()
                    outputs2 = np.argmax(logits2, axis=1)

                    logits3 = logits3.detach().cpu().numpy()
                    outputs3 = np.argmax(logits3, axis=1)

                    logits4 = logits4.detach().cpu().numpy()
                    outputs4 = np.argmax(logits4, axis=1)

                    outputs = []

                    # Next, use voting technology to calculate the current outputs
                    for i in range(len(outputs1)):
                        value = [outputs1[i], outputs2[i], outputs3[i], outputs4[i]]
                        count_of_1 = value.count(1)
                        
                        if count_of_1 >= 2:
                            outputs.append(1)
                        else:
                            outputs.append(0)
                    predict_list.extend(outputs)
                    print(outputs)
                    for output in outputs:
                        f.write(str(output) + "\n")
                    

    import pandas as pd

    data = pd.read_csv('unclassified_files/'+args.data_dir+'.csv')
    
    data['predict'] = predict_list
    data.to_csv( args.output_dir + '/predict_'+args.data_dir+'.csv', index=False)



if __name__ == "__main__":
    main()                           
