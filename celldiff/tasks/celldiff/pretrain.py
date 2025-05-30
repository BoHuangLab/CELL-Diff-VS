# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from celldiff.data.vscyto3d_zarr_data.dataset import VSCyto3DZARRDataset
from celldiff.models.celldiff.config import CELLDiffLD3DVSConfig
from celldiff.models.celldiff.model import CELLDiffLD3DVSModel
from celldiff.utils.cli_utils import cli
from transformers import Trainer, TrainingArguments
from celldiff.logging.loggers import VanillaLoggingCallback
from celldiff.logging import logger

@cli(CELLDiffLD3DVSConfig)
def main(args) -> None:

    config = CELLDiffLD3DVSConfig(**vars(args))

    trainset = VSCyto3DZARRDataset(config, split_key=config.split_key)
    model = CELLDiffLD3DVSModel(config=config)

    logger.info(config)

    training_args = TrainingArguments(
        output_dir=args.output_dir, 
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay, 
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        per_device_train_batch_size=args.per_device_train_batch_size, 
        per_device_eval_batch_size=args.per_device_eval_batch_size, 
        num_train_epochs=args.num_train_epochs, 
        fp16=args.fp16, 
        bf16=args.bf16, 
        logging_dir=args.logging_dir, 
        logging_steps=args.logging_steps, 
        max_steps=args.max_steps, 
        warmup_steps=args.warmup_steps, 
        save_steps=args.save_steps, 
        seed=args.seed, 
        dataloader_num_workers=args.dataloader_num_workers, 
        report_to='wandb', 
        disable_tqdm=True, 
        remove_unused_columns=False, 
        overwrite_output_dir=True, 
        log_level='debug', 
        include_inputs_for_metrics=False, 
        save_safetensors=False, 
    )

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=trainset, 
        data_collator=trainset.collate, 
        callbacks=[VanillaLoggingCallback()], 
    )

    trainer.train(resume_from_checkpoint=args.ifresume)

if __name__ == "__main__":
    main()

