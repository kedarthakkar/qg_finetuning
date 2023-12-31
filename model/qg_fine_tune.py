from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
import torch
import numpy as np
from typing import Optional, List


class Seq2SeqFineTune:
    # TODO: Type annotation for metric
    def __init__(
        self,
        dataset_name: str,
        train_sample_key: str = "content",
        train_label_key: str = "target",
        base_checkpoint: str = "facebook/bart-large-cnn",
        checkpoint_dir: str = "checkpoint_dir",
        tokenizer=None,
        max_input_length: int = 1024,
        max_target_length: int = 128,
        model_filepath: str = "fairytale_qg",
        metric_name: str = "rouge",
        seed: int = 334,
    ):
        self.dataset_name = dataset_name
        self.train_sample_key = train_sample_key
        self.train_label_key = train_label_key
        self.base_checkpoint = base_checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.model_filepath = model_filepath
        self.metric = load_metric(metric_name)
        self.seed = seed

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_checkpoint)
        else:
            self.tokenizer = tokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Add function type annotation
    def load_datasets(self, splits: List[str], sample_size: Optional[List[str]] = None):
        if sample_size is not None:
            if len(splits) != len(sample_size):
                raise ValueError(
                    "Sample size list must be the same length as splits list if provided."
                )

        dataset_list = [
            load_dataset(self.dataset_name, split=curr_split) for curr_split in splits
        ]
        if sample_size is not None:
            dataset_list = [
                dataset_list[i].shuffle(seed=self.seed).select(range(sample_size[i]))
                for i in range(len(dataset_list))
            ]

        return dataset_list

    def tokenize_datasets(self, dataset_list):
        tokenized_datasets = [
            dataset.map(
                self._tokenize_function,
                batched=True,
                remove_columns=[
                    "story_name",
                    "content",
                    "answer",
                    "question",
                    "gem_id",
                    "target",
                    "references",
                    "local_or_sum",
                    "attribute",
                    "ex_or_im",
                ],
            )
            for dataset in dataset_list
        ]

        return tokenized_datasets

    def _tokenize_function(self, examples):
        model_inputs = self.tokenizer(
            examples[self.train_sample_key],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Tokenize targets if they exist
        if self.train_label_key in examples:
            with self.tokenizer.as_target_tokenizer():
                targets = self.tokenizer(
                    examples[self.train_label_key],
                    max_length=self.max_target_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

            model_inputs["labels"] = targets["input_ids"]
        return model_inputs

    # Source: https://github.com/AldoF95/bart-chat-summarizer-finetuning/blob/main/Bart_large_xsum_fine_tuned_samsum.ipynb
    def _compute_rouge(self, pred):
        predictions, labels = pred
        decode_predictions = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decode_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # compute results
        res = self.metric.compute(
            predictions=decode_predictions, references=decode_labels, use_stemmer=True
        )
        res = {key: value.mid.fmeasure * 100 for key, value in res.items()}

        pred_lens = [
            np.count_nonzero(pred != self.tokenizer.pad_token_id)
            for pred in predictions
        ]
        res["gen_len"] = np.mean(pred_lens)
        return {k: round(v, 4) for k, v in res.items()}

    def train(self):
        # Load datasets
        dataset_list = self.load_datasets(splits=["train", "validation"])
        # Tokenize datasets
        tokenized_train_dataset, tokenized_val_dataset = self.tokenize_datasets(
            dataset_list
        )
        tokenized_train_dataset.set_format("torch")
        tokenized_val_dataset.set_format("torch")

        # Initialize model and training arguments
        model = AutoModelForSeq2SeqLM.from_pretrained(self.base_checkpoint)
        model = model.to(self.device)

        collator = DataCollatorForSeq2Seq(self.tokenizer, model=model)
        args = Seq2SeqTrainingArguments(
            self.checkpoint_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=2,
            weight_decay=0.01,
            save_total_limit=2,
            num_train_epochs=3,
            predict_with_generate=True,
            eval_accumulation_steps=1,
            fp16=True,
            load_best_model_at_end=True,
        )

        trainer = Seq2SeqTrainer(
            model,
            args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            data_collator=collator,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_rouge,
        )
        # Train the model
        trainer.train()
        # Save the model
        trainer.save_model(self.model_filepath)

    def batch_infer(self, dataset, batch_size):
        if self.model_filepath is None:
            raise ValueError("Model must be trained before running inference!")

        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_filepath)
        model = model.to(self.device)
        tokenized_dataset = dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=[
                "story_name",
                "content",
                "answer",
                "question",
                "gem_id",
                "target",
                "references",
                "local_or_sum",
                "attribute",
                "ex_or_im",
            ],
        )
        tokenized_dataset.set_format("torch")
        decoded_out_list = []
        for i in np.arange(0, len(tokenized_dataset), batch_size):
            endpoint = min(i + batch_size, len(tokenized_dataset))
            tokenized_out = model.generate(
                tokenized_dataset["input_ids"][i:endpoint].to(self.device),
                num_beams=2,
                min_length=0,
                max_length=50,
            )
            decoded_out = self.tokenizer.batch_decode(
                tokenized_out, skip_special_tokens=True
            )
            decoded_out_list.extend(decoded_out)
            if "labels" in tokenized_dataset.features:
                decoded_labels = self.tokenizer.batch_decode(
                    tokenized_dataset["labels"][i:endpoint], skip_special_tokens=True
                )
                self.metric.add_batch(
                    predictions=decoded_out, references=decoded_labels
                )

        eval_scores = None
        if "labels" in tokenized_dataset.features:
            eval_scores = self.metric.compute()

        return decoded_out_list, eval_scores

    # TODO: Optimize by loading model only once
    def infer(self, example):
        if self.model_filepath is None:
            raise ValueError("Model must be trained before running inference!")

        example_wrapped = {self.train_sample_key: example}
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_filepath)
        model = model.to(self.device)
        tokenized_example = self._tokenize_function(example_wrapped)
        tokenized_out = model.generate(
            tokenized_example["input_ids"].to(self.device),
            num_beams=2,
            min_length=0,
            max_length=50,
        )
        decoded_out = self.tokenizer.batch_decode(
            tokenized_out, skip_special_tokens=True
        )
        return decoded_out
