# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""
import argparse
import glob
import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMTokenizer,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features
)
from modeling_bert_old_v2 import BertForQuestionAnswering_v2

from transformers import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from tokenization_kobert import KoBertTokenizer
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
#from squad_jylee import (SquadResult, SquadV1Processor, SquadV2Processor, 
#    squad_convert_examples_to_features_v2)
from utils import to_list, set_seed, create_logger
from data_to_korquality import data_to_input
from model_to_output import pred_to_output

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


#os.environ['WORK_DIR'] = "/workspace/KoBERT-KorQuAD/prediction-os-v2"
os.environ['WORK_DIR'] = "/code/"

version_2_with_negative = True
do_lower_case = False
max_seq_length = 512
doc_stride = 128
max_query_length = 64
threads = 1
BATCH_SIZE = 16
n_best_size = 20
max_answer_length = 30
verbose_logging = False
null_score_diff_threshold = 0.0
model_type = "kobert"
seed=42

def load_and_cache_examples(args, tokenizer, output_examples=False):
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.tmp_dir, f"cached_{args.korquality_data.split('.')[0]}_{max_seq_length}",
    )

    # Init features and dataset from cache
    #logger.info("Creating features from dataset file at %s", args.data_dir)

    processor = SquadV2Processor() if version_2_with_negative else SquadV1Processor()
    examples = processor.get_dev_examples(args.tmp_dir, filename=args.korquality_data)

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False,
        return_dataset="pt",
        threads=threads,
    )

    #logger.info("Saving features into cached file %s", cached_features_file)
    torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if output_examples:
        return dataset, examples, features
    
    return dataset

def inference(args, model, tokenizer):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, output_examples=True)

    # Note that DistributedSampler samples randomly
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE)
    
    #logger.info("***** Running inference *****")
    all_results = []
    for batch in tqdm(dataloader, desc="Inference"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if model_type in ["xlm", "roberta", "distilbert", "distilkobert"]:
                del inputs["token_type_ids"]

            example_indices = batch[3]
            '''
            # XLNet and XLM use more arguments for their predictions
            if model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )
            '''

            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits, has_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)
            
    # Compute predictions
    output_prediction_file = os.path.join(args.tmp_dir, f"predictions_{args.doc_id}.json")
    output_nbest_file = os.path.join(args.tmp_dir, f"nbest_predictions_{args.doc_id}.json")
    
    if version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.tmp_dir, f"null_odds_{args.doc_id}.json")
    else:
        output_null_log_odds_file = None
    
    # XLNet and XLM use a more complex post-processing procedure
    if model_type in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            n_best_size,
            max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            version_2_with_negative,
            tokenizer,
            args.verbose_logging,
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            n_best_size,
            max_answer_length,
            do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            verbose_logging,
            version_2_with_negative,
            null_score_diff_threshold,
            tokenizer,
        )
    return output_prediction_file

def main(args):
    logger = create_logger()

    ALL_MODELS = sum(
        (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, DistilBertConfig, RobertaConfig, XLNetConfig, XLMConfig)),
        (),
    )
    MODEL_CLASSES = {
        "roberta": (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
        "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
        "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
        "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
        "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
        "kobert": (BertConfig, BertForQuestionAnswering_v2, KoBertTokenizer),
        "distilkobert": (DistilBertConfig, DistilBertForQuestionAnswering, KoBertTokenizer),
    }

    # Setup CGPU
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    args.device = device
    # Set seed
    set_seed(seed)
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    # Load a trained model and vocabulary that you have fine-tuned
    
    config = config_class.from_pretrained(os.path.join(args.model_dir, "config.json"))
    model = model_class.from_pretrained(args.model_dir)  # , force_download=True)
    tokenizer = tokenizer_class.from_pretrained(args.model_dir, do_lower_case=do_lower_case)
    model.to(args.device)
    
    logger.info("Convert to KorQaulity format.")
    args.korquality_data = data_to_input(args)
    
    logger.info("Inference")
    prediction_file = inference(args, model, tokenizer)
    args.prediction_file = prediction_file
    
    args.processing = False
    logger.info("Create pre-trained model output.")
    output_path = pred_to_output(args)
    
    return output_path


if __name__ == "__main__":
    main(args)