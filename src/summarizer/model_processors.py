import json
from collections import OrderedDict
from typing import List, Tuple, Union

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, MT5ForConditionalGeneration, BertModel, MT5EncoderModel, \
    MT5Config, BertConfig

from summarizer.bert_parent import BertParent
from summarizer.cluster_features import ClusterFeatures


class ModelProcessor(object):
    aggregate_map = {
        'mean': np.mean,
        'min': np.min,
        'median': np.median,
        'max': np.max,
    }

    def __init__(
            self,
            model: str = 'mt5',
            custom_model: PreTrainedModel = None,
            custom_tokenizer: PreTrainedTokenizer = None,
            hidden: Union[List[int], int] = -2,
            device: str = 'cpu',
            reduce_option: str = 'mean',
            random_state: int = 666,
            hidden_concat: bool = False,
            gpu_id: int = 0,
    ):
        """
        This is the parent Bert Summarizer model. New methods should implement this class.

        :param model: This parameter is associated with the inherit string parameters from the transformers library.
        :param custom_model: If you have a pre-trained model, you can add the model class here.
        :param custom_tokenizer: If you have a custom tokenizer, you can add the tokenizer here.
        :param hidden: This signifies which layer(s) of the BERT model you would like to use as embeddings.
        :param reduce_option: Given the output of the bert model, this param determines how you want to reduce results.
        CoreferenceHandler instance
        :param random_state: The random state to reproduce summarizations.
        :param hidden_concat: Whether or not to concat multiple hidden layers.
        :param gpu_id: GPU device index if CUDA is available. 
        """
        np.random.seed(random_state)
        # self.model = BertParent(model, custom_model, custom_tokenizer, gpu_id)
        self.model = custom_model
        self.tokenizer = custom_tokenizer
        # self.hidden = hidden
        self.device = device
        self.reduce_option = reduce_option
        self.random_state = random_state
        self.hidden_concat = hidden_concat

    def cluster_runner(
            self,
            content: List[str],
            ratio: float = 0.2,
            algorithm: str = 'kmeans',
            use_first: bool = True,
            num_sentences: int = None
    ) -> Tuple[List[str], np.ndarray]:
        """
        Runs the cluster algorithm based on the hidden state. Returns both the embeddings and sentences.

        :param content: Content list of sentences.
        :param ratio: The ratio to use for clustering.
        :param algorithm: Type of algorithm to use for clustering.
        :param use_first: Return the first sentence in the output (helpful for news stories, etc).
        :param num_sentences: Number of sentences to use for summarization.
        :return: A tuple of summarized sentences and embeddings
        """
        if num_sentences is not None:
            num_sentences = num_sentences if use_first else num_sentences

        src_tokens = [self.tokenizer.tokenize(sent) for sent in content]
        src_token_idxs = [self.tokenizer.convert_tokens_to_ids(sent) for sent in src_tokens]
        src_token_idxs[-1] = src_token_idxs[-1] + [2]
        src_token_idxs = [[1] + sent for sent in src_token_idxs]

        src = torch.tensor(sum(src_token_idxs, []), device=self.device, requires_grad=False).unsqueeze(0)
        clss = torch.tensor([i for i, t in enumerate(sum(src_token_idxs, [])) if t == 1], device=self.device,
                            requires_grad=False).unsqueeze(0)
        mask_src = ~(src == 0)

        hidden = self.model(src, attention_mask=mask_src).last_hidden_state
        hidden = hidden[torch.arange(hidden.size(0)).unsqueeze(1), clss]
        hidden = hidden.detach().numpy()[0]

        hidden_args = ClusterFeatures(
            hidden, algorithm, random_state=self.random_state).cluster(ratio, num_sentences)
        print('src:', len(content), 'ext:', len(hidden_args), hidden_args)

        if use_first:

            if not hidden_args:
                hidden_args.append(0)

            elif hidden_args[0] != 0:
                hidden_args.insert(0, 0)

        sentences = [content[j] for j in hidden_args]
        embeddings = np.asarray([hidden[j] for j in hidden_args])

        return sentences, embeddings

    def __run_clusters(
            self,
            content: List[str],
            ratio: float = 0.2,
            algorithm: str = 'kmeans',
            use_first: bool = True,
            num_sentences: int = None
    ) -> List[str]:
        """
        Runs clusters and returns sentences.

        :param content: The content of sentences.
        :param ratio: Ratio to use for for clustering.
        :param algorithm: Algorithm selection for clustering.
        :param use_first: Whether to use first sentence
        :param num_sentences: Number of sentences. Overrides ratio.
        :return: summarized sentences
        """
        sentences, _ = self.cluster_runner(
            content, ratio, algorithm, use_first, num_sentences)
        return sentences

    def __retrieve_summarized_embeddings(
            self,
            content: List[str],
            ratio: float = 0.2,
            algorithm: str = 'kmeans',
            use_first: bool = True,
            num_sentences: int = None
    ) -> np.ndarray:
        """
        Retrieves embeddings of the summarized sentences.

        :param content: The content of sentences.
        :param ratio: Ratio to use for for clustering.
        :param algorithm: Algorithm selection for clustering.
        :param use_first: Whether to use first sentence
        :return: Summarized embeddings
        """
        _, embeddings = self.cluster_runner(
            content, ratio, algorithm, use_first, num_sentences)
        return embeddings

    def calculate_elbow(
            self,
            body: List[str],
            algorithm: str = 'kmeans',
            k_max: int = None,
    ) -> List[float]:
        """
        Calculates elbow across the clusters.

        :param body: The input body to summarize.
        :param algorithm: The algorithm to use for clustering.
        :param min_length: The min length to use.
        :param max_length: The max length to use.
        :param k_max: The maximum number of clusters to search.
        :return: List of elbow inertia values.
        """
        sentences = body

        if k_max is None:
            k_max = len(sentences) - 1

        hidden = self.model(sentences, self.hidden,
                            self.reduce_option, hidden_concat=self.hidden_concat)
        elbow = ClusterFeatures(
            hidden, algorithm, random_state=self.random_state).calculate_elbow(k_max)

        return elbow

    def calculate_optimal_k(
            self,
            body: List[str],
            algorithm: str = 'kmeans',
            k_max: int = None,
    ):
        """
        Calculates the optimal Elbow K.

        :param body: The input body to summarize.
        :param algorithm: The algorithm to use for clustering.
        :param k_max: The maximum number of clusters to search.
        :return:
        """
        sentences = body

        if k_max is None:
            k_max = len(sentences) - 1

        hidden = self.model(sentences, self.hidden,
                            self.reduce_option, hidden_concat=self.hidden_concat)
        optimal_k = ClusterFeatures(
            hidden, algorithm, random_state=self.random_state).calculate_optimal_cluster(k_max)

        return optimal_k

    def run(
            self,
            body: List[str],
            ratio: float = 0.2,
            use_first: bool = True,
            algorithm: str = 'kmeans',
            num_sentences: int = None,
            return_as_list: bool = False
    ) -> Union[List, str]:
        """
        Preprocesses the sentences, runs the clusters to find the centroids, then combines the sentences.

        :param body: The raw string body to process
        :param ratio: Ratio of sentences to use
        :param min_length: Minimum length of sentence candidates to utilize for the summary.
        :param max_length: Maximum length of sentence candidates to utilize for the summary
        :param use_first: Whether or not to use the first sentence
        :param algorithm: Which clustering algorithm to use. (kmeans, gmm)
        :param num_sentences: Number of sentences to use (overrides ratio).
        :param return_as_list: Whether or not to return sentences as list.
        :return: A summary sentence
        """
        # sentences = self.sentence_handler(body, min_length, max_length)
        sentences = body

        if sentences:
            sentences = self.__run_clusters(
                sentences, ratio, algorithm, use_first, num_sentences)

        if return_as_list:
            return sentences
        else:
            return ' '.join(sentences)

    def __call__(
            self,
            body: List[str],
            ratio: float = 0.2,
            use_first: bool = True,
            algorithm: str = 'kmeans',
            num_sentences: int = None,
            return_as_list: bool = False,
    ) -> str:
        """
        (utility that wraps around the run function)
        Preprocesses the sentences, runs the clusters to find the centroids, then combines the sentences.

        :param body: The raw string body to process.
        :param ratio: Ratio of sentences to use.
        :param use_first: Whether or not to use the first sentence.
        :param algorithm: Which clustering algorithm to use. (kmeans, gmm)
        :param Number of sentences to use (overrides ratio).
        :param return_as_list: Whether or not to return sentences as list.
        :return: A summary sentence.
        """
        return self.run(
            body, ratio, algorithm=algorithm, use_first=use_first, num_sentences=num_sentences,
            return_as_list=return_as_list
        )


class Summarizer(ModelProcessor):

    def __init__(
            self,
            model: str = 'bert-large-uncased',
            custom_model: PreTrainedModel = None,
            custom_tokenizer: PreTrainedTokenizer = None,
            hidden: Union[List[int], int] = -2,
            device: str = 'cpu',
            reduce_option: str = 'mean',
            random_state: int = 666,
            hidden_concat: bool = False,
            gpu_id: int = 0,
    ):
        """
        This is the main Bert Summarizer class.

        :param model: This parameter is associated with the inherit string parameters from the transformers library.
        :param custom_model: If you have a pre-trained model, you can add the model class here.
        :param custom_tokenizer: If you have a custom tokenizer, you can add the tokenizer here.
        :param hidden: This signifies which layer of the BERT model you would like to use as embeddings.
        :param reduce_option: Given the output of the bert model, this param determines how you want to reduce results.
        :param random_state: The random state to reproduce summarizations.
        :param hidden_concat: Whether or not to concat multiple hidden layers.
        :param gpu_id: GPU device index if CUDA is available. 
        """

        super(Summarizer, self).__init__(
            model, custom_model, custom_tokenizer, hidden, device, reduce_option, random_state, hidden_concat, gpu_id
        )

# class TransformerSummarizer(ModelProcessor):
#     """
#     Newer style that has keywords for models and tokenizers, but allows the user to change the type.
#     """
#
#     MODEL_DICT = {
#         'Bert': (BertModel, BertTokenizer),
#         'OpenAIGPT': (OpenAIGPTModel, OpenAIGPTTokenizer),
#         'GPT2': (GPT2Model, GPT2Tokenizer),
#         'CTRL': (CTRLModel, CTRLTokenizer),
#         'TransfoXL': (TransfoXLModel, TransfoXLTokenizer),
#         'XLNet': (XLNetModel, XLNetTokenizer),
#         'XLM': (XLMModel, XLMTokenizer),
#         'DistilBert': (DistilBertModel, DistilBertTokenizer),
#     }
#
#     def __init__(
#         self,
#         transformer_type: str = 'Bert',
#         transformer_model_key: str = 'bert-base-uncased',
#         transformer_tokenizer_key: str = None,
#         hidden: Union[List[int], int] = -2,
#         reduce_option: str = 'mean',
#         random_state: int = 12345,
#         hidden_concat: bool = False,
#         gpu_id: int = 0,
#     ):
#         """
#         :param transformer_type: The Transformer type, such as Bert, GPT2, DistilBert, etc.
#         :param transformer_model_key: The transformer model key. This is the directory for the model.
#         :param transformer_tokenizer_key: The transformer tokenizer key. This is the tokenizer directory.
#         :param hidden: The hidden output layers to use for the summarization.
#         :param reduce_option: The reduce option, such as mean, max, min, median, etc.
#         :param sentence_handler: The sentence handler class to process the raw text.
#         :param random_state: The random state to use.
#         :param hidden_concat: Deprecated hidden concat option.
#         :param gpu_id: GPU device index if CUDA is available.
#         """
#         try:
#             self.MODEL_DICT['Roberta'] = (RobertaModel, RobertaTokenizer)
#             self.MODEL_DICT['Albert'] = (AlbertModel, AlbertTokenizer)
#             self.MODEL_DICT['Camembert'] = (CamembertModel, CamembertTokenizer)
#             self.MODEL_DICT['Bart'] = (BartModel, BartTokenizer)
#             self.MODEL_DICT['Longformer'] = (LongformerModel, LongformerTokenizer)
#         except Exception:
#             pass  # older transformer version
#
#         model_clz, tokenizer_clz = self.MODEL_DICT[transformer_type]
#         model = model_clz.from_pretrained(
#             transformer_model_key, output_hidden_states=True)
#
#         tokenizer = tokenizer_clz.from_pretrained(
#             transformer_tokenizer_key if transformer_tokenizer_key is not None else transformer_model_key
#         )
#
#         super().__init__(
#             None, model, tokenizer, hidden, reduce_option, random_state, hidden_concat, gpu_id
#         )
