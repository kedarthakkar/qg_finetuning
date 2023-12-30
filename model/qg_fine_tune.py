from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
import torch
import numpy as np
from typing import Optional


class QGFineTune:
    # TODO: Type annotation for metric
    def __init__(
        self,
        base_checkpoint: str = "facebook/bart-large-cnn",
        tokenizer=None,
        max_input_length: int = 1024,
        max_target_length: int = 128,
        model_filepath: str = "fairytale_qg",
        metric=load_metric("rouge"),
        seed: int = 334,
        train_sample_size: Optional[int] = None,
        val_sample_size: Optional[int] = None,
        test_sample_size: Optional[int] = None,
    ):
        self.base_checkpoint = base_checkpoint
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.model_filepath = model_filepath
        self.metric = metric
        self.seed = seed
        self.train_sample_size = train_sample_size
        self.val_sample_size = val_sample_size
        self.test_sample_size = test_sample_size

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_checkpoint)
        else:
            self.tokenizer = tokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Add function type annotation
    def load_datasets(self):
        train_dataset = load_dataset("GEM/FairytaleQA", split="train")
        val_dataset = load_dataset("GEM/FairytaleQA", split="validation")
        test_dataset = load_dataset("GEM/FairytaleQA", split="test")

        if self.train_sample_size:
            train_dataset = train_dataset.shuffle(seed=self.seed).select(
                range(self.train_sample_size)
            )

        if self.val_sample_size:
            val_dataset = val_dataset.shuffle(seed=self.seed).select(
                range(self.val_sample_size)
            )

        if self.test_sample_size:
            test_dataset = test_dataset.shuffle(seed=self.seed).select(
                range(self.test_sample_size)
            )

        return train_dataset, val_dataset, test_dataset

    def tokenize_datasets(self, train_dataset, val_dataset, test_dataset):
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
            for dataset in [train_dataset, val_dataset, test_dataset]
        ]

        return tokenized_datasets[0], tokenized_datasets[1], tokenized_datasets[2]

    def _tokenize_function(self, examples):
        model_inputs = self.tokenizer(
            examples["content"],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Tokenize targets if they exist
        if 'target' in examples:
            with self.tokenizer.as_target_tokenizer():
                targets = self.tokenizer(
                    examples["target"],
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
        train_dataset, val_dataset, test_dataset = self.load_datasets()
        # Tokenize datasets
        tokenized_train_dataset, tokenized_val_dataset, _ = self.tokenize_datasets(
            train_dataset, val_dataset, test_dataset
        )
        # Initialize model and training arguments
        model = AutoModelForSeq2SeqLM.from_pretrained(self.base_checkpoint)
        model = model.to(self.device)

        collator = DataCollatorForSeq2Seq(self.tokenizer, model=model)
        args = Seq2SeqTrainingArguments(
            "fairytale-qg-test",
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

    def infer_dataset(self, dataset):
        if self.model_filepath is None:
            raise ValueError("Model must be trained before running inference!")

        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_filepath)
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
        tokenized_out = model.generate(
            tokenized_dataset["input_ids"].to(self.device),
            num_beams=2,
            min_length=0,
            max_length=50,
        )
        decoded_out = self.tokenizer.batch_decode(
            tokenized_out, skip_special_tokens=True
        )
        return decoded_out

    # TODO: Optimize by loading model only once
    def infer(self, example):
        if self.model_filepath is None:
            raise ValueError("Model must be trained before running inference!")

        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_filepath)
        tokenized_example = self._tokenize_function(example)
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