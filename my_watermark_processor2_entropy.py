"""
Entropy-based Filtering:

Key features:
- Computes z-scores and p-values to assess watermark presence.
- Supports entropy-based filtering of tokens and bigram-level analysis.
- Optionally ignores repeated bigrams during scoring.
- Returns detailed results including token counts, green token fractions, prediction flags, and green masks.


Author:
Jingmiao Li
"""


from __future__ import annotations
from math import sqrt

import scipy.stats

import torch
from torch import Tensor
from tokenizers import Tokenizer
from transformers import LogitsProcessor
import collections
from nltk.util import ngrams
from normalizers import normalization_strategy_lookup


class WatermarkBase:
    def __init__(
        self,
        vocab: list[int] = None,
        gamma: float = 0.5,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",  # mostly unused/always default
        hash_key: int = 15485863,  # just a large prime number to create a rng seed with sufficient bit width
        select_green_tokens: bool = True,
    ):

        # watermarking parameters
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gamma = gamma
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        self.rng = None
        self.hash_key = hash_key
        self.select_green_tokens = select_green_tokens

    def _seed_rng(self, input_ids: torch.LongTensor, seeding_scheme: str = None) -> None:
        # can optionally override the seeding scheme,
        # but uses the instance attr by default
        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme

        if seeding_scheme == "simple_1":
            assert input_ids.shape[-1] >= 1, f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed rng"
            prev_token = input_ids[-1].item()
            self.rng.manual_seed(self.hash_key * prev_token)
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")
        return

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        # seed the rng using the previous tokens/prefix
        # according to the seeding_scheme
        self._seed_rng(input_ids)

        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.rng)
        if self.select_green_tokens:  # directly
            greenlist_ids = vocab_permutation[:greenlist_size]  # new
        else:  # select green via red
            greenlist_ids = vocab_permutation[(self.vocab_size - greenlist_size) :]  # legacy behavior
        return greenlist_ids


class WatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        # TODO lets see if we can lose this loop
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        # this is lazy to allow us to colocate on the watermarked model's device
        if self.rng is None:
            self.rng = torch.Generator(device=input_ids.device)

        # NOTE, it would be nice to get rid of this batch loop, but currently,
        # the seed and partition operations are not tensor/vectorized, thus
        # each sequence in the batch needs to be treated separately.
        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self._get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta)
        return scores


class WatermarkDetector(WatermarkBase):
    def __init__(
        self,
        *args,
        device: torch.device = None,
        tokenizer: Tokenizer = None,
        z_threshold: float = 4.0,
        normalizers: list[str] = ["unicode"],  # or also: ["unicode", "homoglyphs", "truecase"]
        ignore_repeated_bigrams: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # also configure the metrics returned/preprocessing options
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"

        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)

        if self.seeding_scheme == "simple_1":
            self.min_prefix_len = 1
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {self.seeding_scheme}")

        self.normalizers = []
        for normalization_strategy in normalizers:
            self.normalizers.append(normalization_strategy_lookup(normalization_strategy))

        self.ignore_repeated_bigrams = ignore_repeated_bigrams
        if self.ignore_repeated_bigrams:
            assert self.seeding_scheme == "simple_1", "No repeated bigram credit variant assumes the single token seeding scheme."

    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        print(f"T: {T}, Observed_count:{observed_count}, Gamma: {self.gamma}, Expected count: {expected_count}, Denominator: {denom}")

        z = numer / denom
        return z

    def _compute_p_value(self, z):
        p_value = scipy.stats.norm.sf(z)
        return p_value

    def _score_sequence(
        self,
        input_ids: Tensor,
        entropy_values: Tensor = None,  
        entropy_threshold: float = None,  
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_p_value: bool = True,
    ):
            
        num_tokens_scored = 0 
        green_token_count = 0  

        if entropy_threshold is not None: 
            # Bigram-level filtering if ignore_repeated_bigrams is True
            if self.ignore_repeated_bigrams:
                # assert return_green_token_mask is False, "Can't return the green/red mask when ignoring repeats."
                print("Warning: Generating bigram_entropy_mask under ignore_repeated_bigrams mode.")

                bigram_list = list(ngrams(input_ids.cpu().tolist(), 2))

                # record unique bigram's idx when first appearing
                unique_bigram_first_idx = {}
                for idx, bigram in enumerate(bigram_list):
                    if bigram not in unique_bigram_first_idx:
                        unique_bigram_first_idx[bigram] = idx

                
                bigram_table = {}
                for bigram, first_idx in unique_bigram_first_idx.items():
                    prefix = torch.tensor([bigram[0]], device=self.device)
                
                    token_entropy_1 = entropy_values[first_idx]
                    
    
                    # skip the bigram if lower than threshold 
                    if token_entropy_1 < entropy_threshold:
                        continue
                    
                    num_tokens_scored += 1
                    greenlist_ids = self._get_greenlist_ids(prefix)
                
                    bigram_table[bigram] = True if bigram[1] in greenlist_ids else False

                green_token_count = sum(bigram_table.values())

            else:
                skip_next_token = False 
                num_tokens_scored = len(input_ids) - self.min_prefix_len  # 初始化为有效的token数量

                for idx in range(self.min_prefix_len, len(input_ids)):
                    if skip_next_token:
                        skip_next_token = (entropy_threshold is not None and entropy_values[idx] < entropy_threshold)
                        print(f"Skipping Token Index: {idx}, Token: {input_ids[idx]}, Entropy: {entropy_values[idx].item()}")
                        # if skip，decrease num_tokens_scored
                        num_tokens_scored -= 1
                        continue

                    # obtain greenlist，and check if the current token is in greenlist
                    greenlist_ids = self._get_greenlist_ids(input_ids[:idx])
                    curr_token = input_ids[idx]
                    if curr_token in greenlist_ids:
                        green_token_count += 1
                    skip_next_token = (entropy_values[idx] < entropy_threshold)

                print(f"Number of Tokens Scored: {num_tokens_scored}")
                print(f"Number of Green Tokens: {green_token_count}")
        else: # no entropy threshold 
            if self.ignore_repeated_bigrams:
                # Method that only counts a green/red hit once per unique bigram.
                # New num total tokens scored (T) becomes the number unique bigrams.
                # We iterate over all unqiue token bigrams in the input, computing the greenlist
                # induced by the first token in each, and then checking whether the second
                # token falls in that greenlist.

                # assert return_green_token_mask is False, "Can't return the green/red mask when ignoring repeats."
                bigram_table = {}
                token_bigram_generator = ngrams(input_ids.cpu().tolist(), 2)
                freq = collections.Counter(token_bigram_generator)
                num_tokens_scored = len(freq.keys())
                for idx, bigram in enumerate(freq.keys()):
                    prefix = torch.tensor([bigram[0]], device=self.device)  # expects a 1-d prefix tensor on the randperm device
                    greenlist_ids = self._get_greenlist_ids(prefix)
                    bigram_table[bigram] = True if bigram[1] in greenlist_ids else False
                green_token_count = sum(bigram_table.values())
            else:
                num_tokens_scored = len(input_ids) - self.min_prefix_len
                if num_tokens_scored < 1:
                    raise ValueError(
                        (
                            f"Must have at least {1} token to score after "
                            f"the first min_prefix_len={self.min_prefix_len} tokens required by the seeding scheme."
                        )
                    )
                # Standard method.
                # Since we generally need at least 1 token (for the simplest scheme)
                # we start the iteration over the token sequence with a minimum
                # num tokens as the first prefix for the seeding scheme,
                # and at each step, compute the greenlist induced by the
                # current prefix and check if the current token falls in the greenlist.
                green_token_count, green_token_mask = 0, []
                for idx in range(self.min_prefix_len, len(input_ids)):
                    curr_token = input_ids[idx]
                    greenlist_ids = self._get_greenlist_ids(input_ids[:idx])
                    if curr_token in greenlist_ids:
                        green_token_count += 1
                        green_token_mask.append(True)
                    else:
                        green_token_mask.append(False)
        # Calculate and return scores
        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=(green_token_count / num_tokens_scored) if num_tokens_scored > 0 else 0))
        if return_z_score:
            score_dict.update(dict(z_score=self._compute_z_score(green_token_count, num_tokens_scored)))
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))

        if return_green_token_mask:
            # token_entropy_mask
            score_dict["token_entropy_mask"] = self.get_token_entropy_triples(input_ids, entropy_values)

            # bigram_entropy_mask
            score_dict["bigram_entropy_mask"] = self.get_bigram_entropy_triples(input_ids, entropy_values)
        
        return score_dict

    def detect(
        self,
        text: str = None,
        tokenized_text: list[int] = None,
        model=None, 
        tokenizer=None,
        device=None, 
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        entropy_threshold = None,  # 可传入单个 float 或 list[float]
        **kwargs,
    ) -> dict:
        """
        If multiple entropy thresholds are provided, a dictionary is returned in the following format:
        {threshold_1: list of detection results, threshold_2: list of detection results, ...}
        Otherwise, a dictionary is returned with a single key corresponding to the provided 
        threshold (for example, None).
        """
        assert (text is not None) ^ (tokenized_text is not None), "Must provide either raw text or pre-tokenized tex"
        if return_prediction:
            kwargs["return_p_value"] = True

        # ----- Normalization -----
        if text is not None:
            if isinstance(text, list):
                normalized_texts = []
                for t in text:
                    for normalizer in self.normalizers:
                        t = normalizer(t)
                    normalized_texts.append(t)
                text = normalized_texts
            else:
                for normalizer in self.normalizers:
                    text = normalizer(text)
                print(f"Text after normalization:\n\n{text}\n")

        # ----- Tokenization and Padding -----
        if tokenized_text is None:
            tokenized_output = self.tokenizer(
                text, return_tensors="pt", add_special_tokens=False, padding=True
            )
            tokenized_text = tokenized_output["input_ids"].to(self.device)
            attention_mask = tokenized_output["attention_mask"].to(self.device)
            if tokenized_text.size(0) == 1:
                # Single sample
                tokenized_text = tokenized_text[0]
                attn_mask = attention_mask[0]
                if tokenized_text[0] == self.tokenizer.bos_token_id:
                    tokenized_text = tokenized_text[1:]
                    attn_mask = attn_mask[1:]
                seq_length = int(attn_mask.sum().item())
                tokenized_text = tokenized_text[:seq_length]
            else:
                # batch samples 
                tokenized_list = []
                for i in range(tokenized_text.size(0)):
                    sample = tokenized_text[i]
                    mask_sample = attention_mask[i]
                    if sample[0] == self.tokenizer.bos_token_id:
                        sample = sample[1:]
                        mask_sample = mask_sample[1:]
                    seq_length = int(mask_sample.sum().item())
                    sample = sample[:seq_length]
                    tokenized_list.append(sample)
                tokenized_text = tokenized_list
        else:
            if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
                tokenized_text = tokenized_text[1:]

        if entropy_threshold is None:
            thresholds = [None]
        elif isinstance(entropy_threshold, list):
            thresholds = entropy_threshold
        else:
            thresholds = [entropy_threshold]

        entropy_values = None
        if model is not None and tokenizer is not None and device is not None:
            with torch.no_grad():
                if isinstance(tokenized_text, list):
                    # tranfer list into padded tensor（only for entropy computing，not affect detection）
                    padded = torch.nn.utils.rnn.pad_sequence(
                        tokenized_text, batch_first=True, padding_value=self.tokenizer.pad_token_id
                    )
                    inputs = {"input_ids": padded}
                else:
                    inputs = {"input_ids": tokenized_text.unsqueeze(0)}
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                entropy_values = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
                if entropy_values.dim() == 2 and entropy_values.size(0) == 1:
                    entropy_values = entropy_values.squeeze(0)
            if isinstance(tokenized_text, list):
                entropy_values = [entropy_values[i] for i in range(entropy_values.size(0))]
            kwargs["entropy_values"] = entropy_values

        results = {}
        if isinstance(tokenized_text, list):
            
            for thr in thresholds:
                sample_results = []
                local_kwargs = kwargs.copy()
                local_kwargs["entropy_threshold"] = thr
                for i in range(len(tokenized_text)):
                    local_kwargs["entropy_values"] = kwargs["entropy_values"][i]
                    sample_score_dict = self._score_sequence(tokenized_text[i], **local_kwargs)
                    output_dict = {}
                    if return_scores:
                        output_dict.update(sample_score_dict)
                    if return_prediction:
                        effective_z_threshold = z_threshold if z_threshold is not None else self.z_threshold
                        assert effective_z_threshold is not None, "Please set a z_threshold"
                        output_dict["prediction"] = sample_score_dict["z_score"] > effective_z_threshold
                        if output_dict["prediction"]:
                            output_dict["confidence"] = 1 - sample_score_dict["p_value"]
                    sample_results.append(output_dict)
                results[thr] = sample_results
            return results
        else:
            
            for thr in thresholds:
                kwargs["entropy_threshold"] = thr
                score_dict = self._score_sequence(tokenized_text, **kwargs)
                output_dict = {}
                if return_scores:
                    output_dict.update(score_dict)
                if return_prediction:
                    effective_z_threshold = z_threshold if z_threshold is not None else self.z_threshold
                    assert effective_z_threshold is not None, "Please set a z_threshold"
                    output_dict["prediction"] = score_dict["z_score"] > effective_z_threshold
                    if output_dict["prediction"]:
                        output_dict["confidence"] = 1 - score_dict["p_value"]
                results[thr] = [output_dict]

            return results


    def get_token_entropy_triples(self, input_ids: Tensor, entropy_values: Tensor) -> list:
        """
        Return a list, with each element:(token, entropy_value, is_green);
        If idx < self.min_prefix_len, set is_green = None.
        """
        token_ids = input_ids.tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        entropy_list = entropy_values.tolist()
        triples = []
        for idx, token in enumerate(tokens):
            current_entropy = float(entropy_list[idx])
            if idx < self.min_prefix_len:
                is_green = None
            else:
                greenlist_ids = self._get_greenlist_ids(input_ids[:idx])
                is_green = (input_ids[idx] in greenlist_ids)
            triples.append((token, current_entropy, is_green))
        return triples

    def get_bigram_entropy_triples(self, input_ids: Tensor, entropy_values: Tensor) -> list:
        """
        Return a list, with each element: ([bigram_pairs], entropy_value, is_green)
        """
        
        token_ids = input_ids.tolist()
        bigram_list = list(ngrams(token_ids, 2))
        unique_bigram_first_idx = {}
        for idx, bigram in enumerate(bigram_list):
            if bigram not in unique_bigram_first_idx:
                unique_bigram_first_idx[bigram] = idx
        triples = []
        for bigram, first_idx in unique_bigram_first_idx.items():
            bigram_tokens = self.tokenizer.convert_ids_to_tokens(list(bigram))
            ent = float(entropy_values[first_idx].item())
            prefix = torch.tensor([bigram[0]], device=self.device)
            # input_ids[:first_idx+1] to ensure accurate idx 
            greenlist_ids = self._get_greenlist_ids(input_ids[:first_idx+1])
            is_green = (bigram[1] in greenlist_ids)
            triples.append((bigram_tokens, ent, is_green))
        return triples  