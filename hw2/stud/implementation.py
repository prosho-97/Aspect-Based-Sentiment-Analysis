import numpy as np
from typing import List, Tuple, Dict

from model import Model
import random

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl


import nltk

from typing import *
from string import punctuation
import re

from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import TokenClassifierOutput
import torch.nn.functional as F


nltk.download('punkt')

BERT_MODEL = 'roberta-large' # Since we have to do NE recognition I chose to set do_lower_case=False
term_tag_values = ['O', 'AT']
polarities = ['positive', 'negative', 'neutral', 'conflict']
N_LAST_HIDDEN_LAYERS = 4

def build_model_b(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements aspect sentiment analysis of the ABSA pipeline.
            b: Aspect sentiment analysis.
    """
    return StudentModel(device, 'b')

def build_model_ab(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements both aspect identification and sentiment analysis of the ABSA pipeline.
            a: Aspect identification.
            b: Aspect sentiment analysis.

    """
    # return RandomBaseline(mode='ab')
    return StudentModel(device, 'ab')
    #raise NotImplementedError

def build_model_cd(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements both aspect identification and sentiment analysis of the ABSA pipeline 
        as well as Category identification and sentiment analysis.
            c: Category identification.
            d: Category sentiment analysis.
    """
    # return RandomBaseline(mode='cd')
    raise NotImplementedError

class RandomBaseline(Model):

    options_sent = [
        ('positive', 793+1794),
        ('negative', 701+638),
        ('neutral',  365+507),
        ('conflict', 39+72),
    ]

    options = [
        (0, 452),
        (1, 1597),
        (2, 821),
        (3, 524),
    ]

    options_cat_n = [
        (1, 2027),
        (2, 402),
        (3, 65),
        (4, 6),
    ]

    options_sent_cat = [
        ('positive', 1801),
        ('negative', 672),
        ('neutral',  411),
        ('conflict', 164),
    ]

    options_cat = [
        ("anecdotes/miscellaneous", 939),
        ("price", 268),
        ("food", 1008),
        ("ambience", 355),
    ]

    def __init__(self, mode = 'b'):

        self._options_sent = [option[0] for option in self.options_sent]
        self._weights_sent = np.array([option[1] for option in self.options_sent])
        self._weights_sent = self._weights_sent / self._weights_sent.sum()

        if mode == 'ab':
            self._options = [option[0] for option in self.options]
            self._weights = np.array([option[1] for option in self.options])
            self._weights = self._weights / self._weights.sum()
        elif mode == 'cd':
            self._options_cat_n = [option[0] for option in self.options_cat_n]
            self._weights_cat_n = np.array([option[1] for option in self.options_cat_n])
            self._weights_cat_n = self._weights_cat_n / self._weights_cat_n.sum()

            self._options_sent_cat = [option[0] for option in self.options_sent_cat]
            self._weights_sent_cat = np.array([option[1] for option in self.options_sent_cat])
            self._weights_sent_cat = self._weights_sent_cat / self._weights_sent_cat.sum()

            self._options_cat = [option[0] for option in self.options_cat]
            self._weights_cat = np.array([option[1] for option in self.options_cat])
            self._weights_cat = self._weights_cat / self._weights_cat.sum()

        self.mode = mode

    def predict(self, samples: List[Dict]) -> List[Dict]:
        preds = []
        for sample in samples:
            pred_sample = {}
            words = None
            if self.mode == 'ab':
                n_preds = np.random.choice(self._options, 1, p=self._weights)[0]
                if n_preds > 0 and len(sample["text"].split(" ")) > n_preds:
                    words = random.sample(sample["text"].split(" "), n_preds)
                elif n_preds > 0:
                    words = sample["text"].split(" ")
            elif self.mode == 'b':
                if len(sample["targets"]) > 0:
                    words = [word[1] for word in sample["targets"]]
            if words:
                pred_sample["targets"] = [(word, str(np.random.choice(self._options_sent, 1, p=self._weights_sent)[0])) for word in words]
            else: 
                pred_sample["targets"] = []
            if self.mode == 'cd':
                n_preds = np.random.choice(self._options_cat_n, 1, p=self._weights_cat_n)[0]
                pred_sample["categories"] = []
                for i in range(n_preds):
                    category = str(np.random.choice(self._options_cat, 1, p=self._weights_cat)[0]) 
                    sentiment = str(np.random.choice(self._options_sent_cat, 1, p=self._weights_sent_cat)[0]) 
                    pred_sample["categories"].append((category, sentiment))
            preds.append(pred_sample)
        return preds


class StudentModel(Model):
    
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def __init__(self, device, mode):

        self.device = device
        self.tokenizer = None
        self.mode = mode

    def spans_A(self, txt: str) -> List[Tuple[int, int]]:
        # This function returns the spans of the several tokens of the input text
        tokens = nltk.word_tokenize(txt)
        tokens = filter(lambda token: self.preprocess_term_A(token) != '', tokens)
        offset = 0
        spans_list = list()
        
        for token in tokens:
            
            if token == '``' or token == "''":
                token = '"'
            elif token[-1] in punctuation and len(token) > 2:
                token = token[:-1]
            if token[0] in punctuation and len(token) > 2:
                token = token[1:]

            filtered_punctuation = punctuation.replace('"', '').replace('.', '').replace("'", '')
            tokens_list = re.split("[" + filtered_punctuation + "]+", token)
            for token in tokens_list:
                if token == '':
                    continue
                offset = txt.find(token, offset)
                spans_list.append((offset, offset+len(token)))
                offset += len(token)
        
        return spans_list



    def preprocess_term_A(self, term: str) -> str: # I remove punctuation
        # This function is used in order to preprocess a signle tokenized term
        cleaned_term = ''
        filtered_punctuation = punctuation.replace('`', '')
        for char in term:
            if (char not in filtered_punctuation) and (char not in '“”'):
                cleaned_term = cleaned_term + char

        return cleaned_term

    def preprocess_text_A(self, text: str) -> List[Dict[str, Union[str, Tuple[int, int]]]]:
        # This function returns a list of dicts, where each dict is associated to a tokenized preprocessed term and contains information about the token itslef, its label and its span indexes

        return list(map(lambda indexes: {'token': text[indexes[0] : indexes[1]], 'indexes': (indexes[0] , indexes[1])}, self.spans_A(text)))

    def encode_text_A(self, sentence:list):
        """
        Args:
            sentences (list): list of Dicts, each carrying the information about
            one token.
        Return:
            The method returns two lists of indexes corresponding to input tokens and span of words that has been splitted into word pieces
        """

        words_pieces_list, word_span_indexes = [self.tokenizer.bos_token], [(0, 1)]
        words_list = [w["token"] for w in sentence]
        index = 1

        for i, word in enumerate(words_list):
          
          tokens = self.tokenizer.tokenize(word, is_split_into_words = i != 0)
          n_word_pieces = len(tokens)

          word_pieces_without_punct = list()
          add_special_symbol = False

          for word_piece_index, word_piece in enumerate(tokens):
            filtered_word_pieces = word_piece.replace('Ġ', '')
            if filtered_word_pieces not in punctuation:
              if add_special_symbol:
                new_word_pieces = self.tokenizer.tokenize(word_piece, is_split_into_words = i != 0)
                word_pieces_without_punct.extend(new_word_pieces)
                add_special_symbol = False
              else:
                word_pieces_without_punct.append(word_piece)
            elif word_piece_index == 0:
              add_special_symbol = True

          n_word_pieces_without_punct = len(word_pieces_without_punct)

          if n_word_pieces_without_punct == 0:
            words_pieces_list.extend(tokens)
            n_word_pieces_without_punct = n_word_pieces
          else:
            words_pieces_list.extend(word_pieces_without_punct)

          word_span_indexes.append((index, index + n_word_pieces_without_punct))
          index += n_word_pieces_without_punct

        words_pieces_list.append(self.tokenizer.eos_token)
        word_span_indexes.append((index, index + 1))

        return torch.LongTensor(self.tokenizer.convert_tokens_to_ids(words_pieces_list)), torch.LongTensor(word_span_indexes)

    def rnn_collate_fn_A(self, data_elements: List[Dict[str, Union[torch.Tensor, List]]]) -> List[Dict[str, Union[torch.Tensor, List]]]:

        X = [de['inputs'] for de in data_elements]
        
        batch = {}
        batch['inputs'] = pad_sequence(X, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch['text'] = [de['text'] for de in data_elements]
        batch['indexes'] = [de['indexes'] for de in data_elements]
        batch['word_span_indexes'] = [de['word_span_indexes'] for de in data_elements]
        batch['attention_masks'] = torch.tensor([[float(i < len(words_span)) for i in range(len(input_ids))] for input_ids, words_span in zip(batch['inputs'], batch['word_span_indexes'])])

        if self.device is not None:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        return batch

    def decode_output_A(self, max_indices:List[List[int]]):
        """
        Args:
            max_indices: a List where the i-th entry is a List containing the
            indexes preds for the i-th sample
        Output:
            The method returns a list of batch_size length where each element is a list
            of labels, one for each input token.
        """
        predictions = list()
        for indices in max_indices:
            predictions.append([term_tag_values[i] for i in indices])
        return predictions

    def get_aspect_terms(self,
                     labels: List[str],
                     text: str,
                     indexes: List[Tuple[int, int]]) -> List[Tuple[str, Tuple[int, int]]]:

        assert len(labels) == len(indexes)
        aspect_terms = []
        start_index = None
        last_seen_aspect_term_appended = False
        filtered_punctuation = punctuation.replace('-', '').replace('"', '').replace('/', '').replace('.', '').replace("'", '').replace('+', '')

        for i in range(len(labels)):

            if labels[i] == 'AT':
                if start_index is None:
                    start_index = indexes[i][0]
                end_index = indexes[i][1]
                last_seen_aspect_term_appended = False

            elif not last_seen_aspect_term_appended and start_index is not None:
                aspect_terms_list = re.split("[" + filtered_punctuation + "]+", text[start_index : end_index])
                start_index_find = start_index
                for aspect_term in aspect_terms_list:
                    if aspect_term != '':
                        aspect_term = aspect_term.strip()
                        aspect_term_start = text.find(aspect_term, start_index_find)
                        aspect_term_len = len(aspect_term)
                        aspect_term_end = aspect_term_start + aspect_term_len
                        assert aspect_term == text[aspect_term_start : aspect_term_end]
                        start_index_find += aspect_term_len
                        aspect_terms.append((aspect_term, (aspect_term_start, aspect_term_end)))
                start_index = None
                last_seen_aspect_term_appended = True

        if not last_seen_aspect_term_appended and start_index is not None:
            aspect_terms_list = re.split("[" + filtered_punctuation + "]+", text[start_index : end_index])
            start_index_find = start_index
            for aspect_term in aspect_terms_list:
                if aspect_term != '':
                    aspect_term = aspect_term.strip()
                    aspect_term_start = text.find(aspect_term, start_index_find)
                    aspect_term_len = len(aspect_term)
                    aspect_term_end = aspect_term_start + aspect_term_len
                    assert aspect_term == text[aspect_term_start : aspect_term_end]
                    start_index_find += aspect_term_len
                    aspect_terms.append((aspect_term, (aspect_term_start, aspect_term_end)))
        
        return aspect_terms

    def evaluate_aspect_terms(self,
                          inputs_indexes: torch.Tensor,
                          decoded_labels: List[List[str]],
                          texts: List[str],
                          batch_indexes: List[List[Tuple[int, int]]]):

        batch_input_tokens = []
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        padding_token_id = self.tokenizer.pad_token_id
        for indexes in inputs_indexes:
            tokens = []
            for index in indexes:
                if index != padding_token_id:
                    if index != bos_token_id and index != eos_token_id:
                        tokens.append(index)
                else:
                    break
            batch_input_tokens.append(self.tokenizer.convert_ids_to_tokens(tokens))

        
        assert len(batch_input_tokens) == len(decoded_labels)
        new_batch_labels = []
        for tokens, labels in zip(batch_input_tokens, decoded_labels):
            labels = labels[1:len(labels) - 1]
            new_labels = []

            label_list_index = 0
            for i, token in enumerate(tokens):
                if token.startswith('Ġ') or i == 0:
                    new_labels.append(labels[label_list_index])
                    label_list_index += 1
            new_batch_labels.append(new_labels)
            assert label_list_index == len(labels)
        
        input_for_B = list()
        assert len(new_batch_labels) == len(texts)
        for i in range(len(new_batch_labels)):
            text = texts[i]
            indexes_list = batch_indexes[i]
            predicted_aspect_terms = self.get_aspect_terms(new_batch_labels[i], text, indexes_list)
            input_for_B.append(predicted_aspect_terms)

        return input_for_B

    def spans_B(self, txt: str) -> List[Tuple[int, int]]:
        # This function returns the spans of the several tokens of the input text
        tokens = nltk.word_tokenize(txt)
        tokens = filter(lambda token: self.preprocess_term_B(token) != '', tokens)
        offset = 0
        spans_list = list()
        for token in tokens:
            offset = txt.find(token, offset)
            spans_list.append((offset, offset+len(token)))
            offset += len(token)
        return spans_list

    def preprocess_term_B(self, term: str) -> str: # I remove punctuation
        # This function is used in order to preprocess a signle tokenized term
        cleaned_term = ''
        for char in term:
            if (char not in punctuation) and (char not in '“”'):
                cleaned_term = cleaned_term + char

        return cleaned_term

    def preprocess_text_B(self, text: str, start_indexes: List[int], end_indexes: List[int]) -> Tuple[List[Dict[str, str]], List[str]]:
        # This function returns a list of dicts, where each dict is associated to a tokenized preprocessed term and contains information about the token itslef and its label
        preprocessed_text = []
        end_indexes.insert(0, 0)
        
        for i in range(len(start_indexes)):
            start_index = start_indexes[i]
            end_index = end_indexes[i + 1]
            preprocessed_text += list(map(lambda indexes: {'token': text[end_indexes[i] + indexes[0] : end_indexes[i] + indexes[1]], 'ne_label': 'O'}, self.spans_B(text[end_indexes[i] : start_index])))
            aspect_term = text[start_index:end_index]
            aspect_term = aspect_term.split(' ')
            aspect_term_list = []
            for j in range(len(aspect_term)):
                term = aspect_term[j]
                preprocessed_term = self.preprocess_term_B(term)
                if preprocessed_term != '':
                    d = {'token': preprocessed_term, 'ne_label': 'B-AT' if len(aspect_term_list) == 0 else 'I-AT'}
                    aspect_term_list.append(d)
                start_index += len(term) + 1
            preprocessed_text += aspect_term_list
        preprocessed_text += list(map(lambda indexes: {'token': text[end_indexes[-1] + indexes[0] : end_indexes[-1] + indexes[1]], 'ne_label': 'O'}, self.spans_B(text[end_indexes[-1] : ])))

        return preprocessed_text

    def rnn_collate_fn_B(self, data_elements: List[Dict[str, Union[torch.Tensor, List]]]) -> List[Dict[str, Union[torch.Tensor, List]]]:

        X = [de['inputs'] for de in data_elements]
        start_indexes, end_indexes = [de['start_indexes'] for de in data_elements], [de['end_indexes'] for de in data_elements] # lists of index tensors
        label_masks = [torch.tensor([1 for _ in de['start_indexes']]) for de in data_elements]
        s = 0
        for elem in start_indexes:
            s += len(elem)
        if s == 0:
            start_indexes, end_indexes = [torch.tensor([0]) for _ in start_indexes], [torch.tensor([0]) for _ in end_indexes]
            label_masks = start_indexes
        
        batch = {}
        batch['inputs'] = pad_sequence(X, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch['label_masks'] = pad_sequence(label_masks, batch_first=True, padding_value=0)
        batch['start_indexes'] = pad_sequence(start_indexes, batch_first=True, padding_value=0)
        batch['end_indexes'] = pad_sequence(end_indexes, batch_first=True, padding_value=0)
        batch['attention_masks'] = torch.tensor([[int(index != self.tokenizer.pad_token_id) for index in input_sample] for input_sample in batch['inputs']])
        batch['aspect_terms'] = [de['aspect_terms'] for de in data_elements]

        assert batch['inputs'].shape == batch['attention_masks'].shape and batch['start_indexes'].shape == batch['end_indexes'].shape == batch['label_masks'].shape


        if self.device is not None:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        return batch

    def get_predicted_output(self, aspect_terms, polarities):
  
        predictions = list()
        
        for aspect_terms_list, polarities_list in zip(aspect_terms, polarities):
            assert len(aspect_terms_list) == len(polarities_list)
            prediction = list()
            for aspect_term, polarity in zip(aspect_terms_list, polarities_list):
                prediction.append((aspect_term, polarity))
            predictions.append({'targets': prediction})

        return predictions

    def encode_text_B(self, sentence:list):

        words_pieces_list, start_indexes, end_indexes = [self.tokenizer.bos_token], list(), list()

        for i, d in enumerate(sentence):

          word = d['token']
          label = d['ne_label']
          
          tokens = self.tokenizer.tokenize(word, is_split_into_words = i != 0) # The Ġ char should not be added at the first word of the sentence

          word_pieces_without_punct = list()
          add_special_symbol = False

          for word_piece_index, word_piece in enumerate(tokens):
            filtered_word_pieces = word_piece.replace('Ġ', '')
            if filtered_word_pieces not in punctuation:
              if add_special_symbol:
                new_word_pieces = self.tokenizer.tokenize(word_piece, is_split_into_words = i != 0)
                word_pieces_without_punct.extend(new_word_pieces)
                add_special_symbol = False
              else:
                word_pieces_without_punct.append(word_piece)
            elif word_piece_index == 0:
              add_special_symbol = True

          if label == 'B-AT':
            start_indexes.append(len(words_pieces_list))

          words_pieces_list.extend(word_pieces_without_punct if len(word_pieces_without_punct) > 0 else tokens)

          if label == 'B-AT':
            end_indexes.append(len(words_pieces_list) - 1)
          elif label == 'I-AT':
            end_indexes[-1] = len(words_pieces_list) - 1
              

        words_pieces_list.append(self.tokenizer.eos_token)

        assert len(start_indexes) == len(end_indexes)

        return torch.LongTensor(self.tokenizer.convert_tokens_to_ids(words_pieces_list)), torch.LongTensor(start_indexes), torch.LongTensor(end_indexes)

    def decode_output_B(self, max_indices:List[List[int]]):
        """
        Args:
            max_indices: a List where the i-th entry is a List containing the
            indexes preds for the i-th sample
        Output:
            The method returns a list of batch_size length where each element is a list
            of labels, one for each input token.
        """
        predictions = list()
        for indices in max_indices:
            predictions.append([polarities[i] for i in indices])
        return predictions

    def get_predictions_for_B(self, batch):
            batch = self.rnn_collate_fn_B(batch)
            
            inputs_indexes = batch['inputs']
            aspect_terms = batch['aspect_terms']
            start_indexes_b = batch['start_indexes']
            label_masks = batch['label_masks']

            inputs = (inputs_indexes, batch['attention_masks'])
            
            logits = self.model_b(inputs, start_indexes_b, batch['end_indexes'])

            batch_size = start_indexes_b.size()[0]
            max_number_of_aspect_terms = start_indexes_b.size()[1]
            sequence_length = logits.size()[1]
            logits = logits.view([batch_size, max_number_of_aspect_terms, sequence_length])

            predictions = torch.argmax(logits, -1)

            predictions = [[int(p_i) for p_i, s_i in zip(p, s) if s_i]
                                    for p, s in zip(predictions, label_masks)]

            decoded_labels = self.decode_output_B(predictions)

            predicted_output = self.get_predicted_output(aspect_terms, decoded_labels)

            return predicted_output


    @torch.no_grad()
    def predict(self, samples: List[Dict]) -> List[Dict]:
        '''
        --> !!! STUDENT: implement here your predict function !!! <--
        Args:
            - If you are doing model_b (ie. aspect sentiment analysis):
                sentence: a dictionary that represents an input sentence as well as the target words (aspects), for example:
                    [
                        {
                            "text": "I love their pasta but I hate their Ananas Pizza.",
                            "targets": [[13, 17], "pasta"], [[36, 47], "Ananas Pizza"]]
                        },
                        {
                            "text": "The people there is so kind and the taste is exceptional, I'll come back for sure!"
                            "targets": [[4, 9], "people", [[36, 40], "taste"]]
                        }
                    ]
            - If you are doing model_ab or model_cd:
                sentence: a dictionary that represents an input sentence, for example:
                    [
                        {
                            "text": "I love their pasta but I hate their Ananas Pizza."
                        },
                        {
                            "text": "The people there is so kind and the taste is exceptional, I'll come back for sure!"
                        }
                    ]
        Returns:
            A List of dictionaries with your predictions:
                - If you are doing target word identification + target polarity classification:
                    [
                        {
                            "targets": [("pasta", "positive"), ("Ananas Pizza", "negative")] # A list having a tuple for each target word
                        },
                        {
                            "targets": [("people", "positive"), ("taste", "positive")] # A list having a tuple for each target word
                        }
                    ]
                - If you are doing target word identification + target polarity classification + aspect category identification + aspect category polarity classification:
                    [
                        {
                            "targets": [("pasta", "positive"), ("Ananas Pizza", "negative")], # A list having a tuple for each target word
                            "categories": [("food", "conflict")]
                        },
                        {
                            "targets": [("people", "positive"), ("taste", "positive")], # A list having a tuple for each target word
                            "categories": [("service", "positive"), ("food", "positive")]
                        }
                    ]
        '''

        if self.tokenizer is None:
            self.model_b = PolarityModule.load_from_checkpoint('model/Model_B.ckpt').model
            self.model_b = self.model_b.to(self.device)
            self.model_b.eval()
            self.tokenizer = RobertaTokenizer.from_pretrained(BERT_MODEL, do_lower_case=False, do_basic_tokenize=False) # I have already done the tokenization with the preprocessing function ---> do_basic_tokenize=False
            if self.mode == 'ab':
                self.model_a = AspectTermModule.load_from_checkpoint('model/Model_A.ckpt').model
                self.model_a = self.model_a.to(self.device)
                self.model_a.eval()

        data, encoded_data = list(), list()

        if self.mode == 'b':

            for sentence_dict in samples:
                indexes_and_aspect_terms = list()
                for target in sentence_dict['targets']:
                    indexes = target[0]
                    indexes_and_aspect_terms.append((indexes[0], indexes[1], target[1]))
                indexes_and_aspect_terms.sort()
                start_indexes, end_indexes, aspect_terms = [elem[0] for elem in indexes_and_aspect_terms], [elem[1] for elem in indexes_and_aspect_terms], [elem[2] for elem in indexes_and_aspect_terms]
                preprocessed_text = self.preprocess_text_B(sentence_dict['text'], start_indexes, end_indexes)
                data.append({"preprocessed_text": preprocessed_text,
                            "aspect_terms": aspect_terms})
            
            for i in range(len(data)):
                # for each sentence
                data_i = data[i]
                elem = data_i["preprocessed_text"]
                encoded_elem, start_indexes, end_indexes = self.encode_text_B(elem)
                encoded_data.append({"inputs": encoded_elem,
                                    "start_indexes": start_indexes,
                                    "end_indexes": end_indexes,
                                    "aspect_terms": data_i['aspect_terms']})

            return self.get_predictions_for_B(encoded_data)

        elif self.mode == 'ab':

            for sentence_dict in samples:
                text = sentence_dict['text']
                preprocessed_text = self.preprocess_text_A(text)
                data.append({"text": text,
                                "preprocessed_text": preprocessed_text})

            for i in range(len(data)):
                # for each sentence
                data_i = data[i]
                elem = data_i["preprocessed_text"]
                encoded_elem, word_span_indexes = self.encode_text_A(elem)
                indexes_list = [w["indexes"] for w in elem]

                encoded_data.append({"inputs": encoded_elem, 
                                        "text": data_i["text"],
                                        "indexes": indexes_list,
                                        "word_span_indexes": word_span_indexes})

            batch = self.rnn_collate_fn_A(encoded_data)

            inputs_indexes = batch['inputs']
            attention_masks = batch['attention_masks']
            inputs = (inputs_indexes, attention_masks)
            logits = self.model_a(inputs, batch['word_span_indexes'])
            predictions = torch.argmax(logits, -1)
            assert predictions.shape == attention_masks.shape
            predictions = [[int(p_i) for p_i, m_i in zip(p, m) if m_i]
                                    for p, m in zip(predictions, attention_masks)]
            decoded_labels = self.decode_output_A(predictions)
            input_for_B = self.evaluate_aspect_terms(inputs_indexes, decoded_labels, batch['text'], batch['indexes'])

            batch_b = list()

            for text, pred in zip(batch['text'], input_for_B):
                start_indexes, end_indexes, aspect_terms = list(), list(), list()
                for (aspect_term, (aspect_term_start, aspect_term_end)) in pred:
                    start_indexes.append(aspect_term_start)
                    end_indexes.append(aspect_term_end)
                    aspect_terms.append(aspect_term)
                assert start_indexes == sorted(start_indexes) and end_indexes == sorted(end_indexes) and len(start_indexes) == len(end_indexes) == len(aspect_terms)
                preprocessed_text = self.preprocess_text_B(text, start_indexes, end_indexes)
                encoded_elem, start_indexes, end_indexes = self.encode_text_B(preprocessed_text)
                assert len(start_indexes) == len(aspect_terms)
                batch_b.append({"inputs": encoded_elem, 
                                "start_indexes": start_indexes,
                                "end_indexes": end_indexes,
                                "aspect_terms": aspect_terms})

            return self.get_predictions_for_B(batch_b)








def get_aspect_terms_word_pieces(start_indexes, end_indexes, contextualized_embs, embs_mask):
    sentences_length = torch.sum(embs_mask, dim=-1)
    terms_offset = torch.cumsum(sentences_length, dim=0)
    terms_offset -= sentences_length
    start_indexes_offset = (start_indexes + terms_offset.unsqueeze(1)).view(-1)
    end_indexes_offset = (end_indexes + terms_offset.unsqueeze(1)).view(-1)

    number_of_word_pieces_per_aspect_term = end_indexes_offset - start_indexes_offset + 1
    max_aspect_term_len = torch.max(number_of_word_pieces_per_aspect_term)
    embs_mask = embs_mask.view(-1)
    batch_size, sequence_length, hidden_dim = contextualized_embs.shape
    contextualized_embs = contextualized_embs.view(batch_size * sequence_length, hidden_dim)
    contextualized_embs = contextualized_embs[embs_mask.nonzero().squeeze(), :]
    text_length = contextualized_embs.shape[0]

    aspect_terms_word_pieces_indexes = torch.arange(max_aspect_term_len).unsqueeze(0).to(start_indexes_offset.device) + start_indexes_offset.unsqueeze(1)
    aspect_terms_word_pieces_indexes = torch.min(aspect_terms_word_pieces_indexes, (text_length - 1) * torch.ones_like(aspect_terms_word_pieces_indexes))
    aspect_terms_word_pieces = contextualized_embs[aspect_terms_word_pieces_indexes, :]
    word_pieces_indexes_range = torch.arange(max_aspect_term_len).to(number_of_word_pieces_per_aspect_term.device)
    aspect_terms_word_pieces_mask = word_pieces_indexes_range < number_of_word_pieces_per_aspect_term.unsqueeze(-1)
    return aspect_terms_word_pieces, aspect_terms_word_pieces_mask.long()

def get_aspect_terms_representation(aspect_terms_word_pieces, attention_scores, aspect_terms_word_pieces_mask):
    aspect_terms_word_pieces_mask = (1.0 - aspect_terms_word_pieces_mask) * -100000.0
    attention_scores = attention_scores + aspect_terms_word_pieces_mask # To avoid to consider padding in the self-attention combination
    probs = (nn.Softmax(dim=-1)(attention_scores)).unsqueeze(-1)
    return torch.sum(probs * aspect_terms_word_pieces, dim=1)

class RobertaForAspectTermClassification(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.a = nn.Linear(config.hidden_size, 1) # It is the a vector named in the report
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
        start_indexes=None,
        end_indexes=None,
        label_masks=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        aspect_terms_word_pieces, aspect_terms_word_pieces_mask = get_aspect_terms_word_pieces(start_indexes, end_indexes, sequence_output, attention_mask)

        attention_scores = (self.a(aspect_terms_word_pieces)).squeeze(-1)
        aspect_terms_representation = get_aspect_terms_representation(aspect_terms_word_pieces, attention_scores, aspect_terms_word_pieces_mask)

        aspect_terms_representation = self.linear(aspect_terms_representation)
        aspect_terms_representation = self.tanh(aspect_terms_representation)
        aspect_terms_representation = self.dropout(aspect_terms_representation)
        logits = self.classifier(aspect_terms_representation)

        loss = None

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class AspectTermClassificationModel(nn.Module):
    # we provide the hyperparameters as input
    def __init__(self, hparams):
        super(AspectTermClassificationModel, self).__init__()

        self.roberta = RobertaForAspectTermClassification.from_pretrained(hparams.bert_model, num_labels=hparams.num_labels)

    
    def forward(self, x, start_indexes, end_indexes):
      b_input_ids, b_input_mask = x
      outputs = self.roberta(b_input_ids, token_type_ids=None,
                          attention_mask=b_input_mask, start_indexes=start_indexes,
                          end_indexes=end_indexes)
      logits = outputs[0]
      return logits

class PolarityModule(pl.LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super(PolarityModule, self).__init__(*args, **kwargs)
        """
        Args:
            model: the model we want to train.
            loss_function: the loss_function to minimize.
            optimizer: the optimizer used to minimize the loss_function.
        """
        self.save_hyperparameters(hparams)
        self.model = AspectTermClassificationModel(self.hparams)

class RobertaForTokenClassificationWithConcat(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear1 = nn.Linear(N_LAST_HIDDEN_LAYERS * config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size//2)
        self.swish = nn.SiLU()
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(config.hidden_size//2, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
        word_span_indexes=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.hidden_states[1:] if return_dict else outputs[2][1:]
        last_layers_concatenated = torch.cat(hidden_states[-N_LAST_HIDDEN_LAYERS:], dim=-1)

        last_layers_concatenated = self.linear1(last_layers_concatenated)
        last_layers_concatenated = self.swish(last_layers_concatenated)

        new_last_layers_concatenated = None
        seq_len = last_layers_concatenated.shape[1]
        for i in range(len(word_span_indexes)):
          words_span = word_span_indexes[i]
          sentence_hidden_states = last_layers_concatenated[i]
          new_sentence_hidden_states = None
          for index1, index2 in words_span:
            word_pieces_mean_hidden_states = torch.mean(sentence_hidden_states[int(index1):int(index2)], dim=0).unsqueeze(0)
            if new_sentence_hidden_states is None:
              new_sentence_hidden_states = word_pieces_mean_hidden_states
            else:
              new_sentence_hidden_states = torch.cat((new_sentence_hidden_states, word_pieces_mean_hidden_states), dim=0)
          new_sentence_hidden_states = F.pad(new_sentence_hidden_states, pad=(0, 0, 0, seq_len - new_sentence_hidden_states.shape[0]))
          new_sentence_hidden_states = new_sentence_hidden_states.unsqueeze(0)
          if new_last_layers_concatenated is None:
            new_last_layers_concatenated = new_sentence_hidden_states
          else:
            new_last_layers_concatenated = torch.cat((new_last_layers_concatenated, new_sentence_hidden_states), dim=0)

        new_last_layers_concatenated = self.dropout(new_last_layers_concatenated)
        new_last_layers_concatenated = self.linear2(new_last_layers_concatenated)
        new_last_layers_concatenated = self.relu(new_last_layers_concatenated)
        new_last_layers_concatenated = self.dropout(new_last_layers_concatenated)
        logits = self.classifier(new_last_layers_concatenated)


        loss = None

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class AspectTermIdentificationModel(nn.Module):
    # we provide the hyperparameters as input
    def __init__(self, hparams):
        super(AspectTermIdentificationModel, self).__init__()

        self.roberta = RobertaForTokenClassificationWithConcat.from_pretrained(hparams.bert_model, num_labels=hparams.num_labels)

    
    def forward(self, x, word_span_indexes):
      b_input_ids, b_input_mask = x
      outputs = self.roberta(b_input_ids, token_type_ids=None,
                          attention_mask=b_input_mask,
                          word_span_indexes=word_span_indexes)
      logits = outputs[0]
      return logits

class AspectTermModule(pl.LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super(AspectTermModule, self).__init__(*args, **kwargs)

        self.save_hyperparameters(hparams)
        self.model = AspectTermIdentificationModel(self.hparams)
