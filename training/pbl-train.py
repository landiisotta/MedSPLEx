"""
Module Title: HPO, Train and Test for PBL 
Author: Eugenia Alleva
Date: 2025-03-24
Description: Train encoder model via prompt-based fine-tuning for MedSPLEx with Hyperparameter Optimization
"""

# === IMPORTS ===
import os
import sys
import argparse
import logging
import itertools
from tqdm import tqdm
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import torch
import yaml
import random
import numpy as np
import pandas as pd
from openprompt.data_utils import InputExample
import copy
from sklearn.metrics import classification_report


# === FUNCTION DEFINITIONS ===
def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Describe the purpose of the script.")
    parser.add_argument("--train", type=str, required=True, help="Path to train dataset")
    parser.add_argument("--val", type=str, required=True, help="Path to validation dataset")
    parser.add_argument("--test", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--best_model_path", type=str, required=True, help="Path to model saved")
    return parser.parse_args()

# === SET SEED ====

torch.manual_seed(43)
torch.cuda.manual_seed(43)
random.seed(43)
np.random.seed(43)


# === CONFIGURATION (Optional) ===
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

with open('config/keywords_updated.yaml') as f:
    config = yaml.safe_load(f) 

# === Hyperparameter Space ===
EPOCHS = 10
param_grid = {
    'learning_rate': [9.5e-6, 1e-5, 1.5e-5, 2e-5, 2.5e-5],
    'batch_size': [1,4,16]
}
# Get all combinations
keys = list(param_grid.keys())
values = list(param_grid.values())
combinations = list(itertools.product(*values))
# Turn each combination into a dictionary
all_param_combos = [dict(zip(keys, combo)) for combo in combinations]


# === Prompt Config ===
# Classes
classes = ["neutral", "stigmatizing", "privileging"]
# Model
plm, tokenizer, model_config, WrapperClass = load_plm("GatorTron", "UFNLP/gatortron-base")
# Template
promptTemplate = ManualTemplate(
    text = '{"placeholder":"text_a"}. This sentence is {"mask"}',
    tokenizer = tokenizer,
)
# Verbalizers
full_verbalizer = ManualVerbalizer(
    classes = classes,
    label_words = {
        "neutral": ["neutral", "impartial", "clinical"], 
        "stigmatizing": ["negative", "stigma", "bad"],
        "privileging": ["positive", "privilege", "good"]
    },
    tokenizer = tokenizer,
)







def main(args):
    """
    Main function where execution starts.
    """
    # === EVALUATION ===
    def eval(dataloader, type='validation'):
        """
        Evaluation.
        """
        ids = []
        labels = []
        predictions = []
        promptModel.eval()
        with torch.no_grad():
            for step,batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Evaluation on {type} dataset'):
                batch.to(DEVICE)
                logits = promptModel(batch)
                preds = torch.argmax(logits, dim=-1)
                predictions = predictions + [int(x) for x in preds]
                labels = labels + [int(x) for x in list(batch['label'])]
        metric = classification_report(labels, predictions, output_dict=True)
        return metric
    # === TRAIN LOOP ===
    def train():
        """
        Training Loop.
        """
        for epoch in tqdm(range(EPOCHS), total=EPOCHS, desc='Full training', position=0):
            promptModel.train()
            epoch_loss = 0    
            for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch+1}', position=1):
                # Prepare batch
                batch = batch.to(DEVICE)
        
                # Forward pass
                logits = promptModel(batch)
                labels = batch['label']
                loss = loss_fun(logits, labels)
        
                # Backpropagation
                loss.backward()
                for opt in optimizers:
                    opt.step()
                    opt.zero_grad()
        
                # Track loss
                epoch_loss += loss.item()
            # evaluate at each epoch
            metrics = eval(val_dataloader)
            logging.info(f'Epoch {epoch}: {metrics}')
            f1 = metrics['macro avg']['f1-score']
            if f1>best_model['F1']:
                logging.info(f"New best model with F1 {f1}")
                best_model['model'] = copy.deepcopy(promptModel.plm)
                best_model['F1'] = f1
                best_model['metrics'] = metrics
                best_model['hpo']=hpo
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Starting Sweep..Working with {DEVICE}")

    # save best performance
    best_model = {'HPO':'hpo', 'F1':0, 'metrics':{}, 'model':'model'}
    
    # Load Datasets
    val_df = pd.read_csv(args.val)
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    
    train_df['label'] = train_df['label'].astype(int)
    val_df['label'] = val_df['label'].astype(int)
    test_df['label'] = test_df['label'].astype(int)

    train_dataset = [InputExample(guid = index,text_a =  f'{x.text}. Keyword is: {x.pattern}.',label=x.label) for index,x in train_df.iterrows()]
    val_dataset = [InputExample(guid = index,text_a =  f'{x.text}. Keyword is: {x.pattern}.',label=x.label) for index,x in val_df.iterrows()]
    test_dataset = [InputExample(guid = index,text_a =  f'{x.text}. Keyword is: {x.pattern}.',label=x.label) for index,x in test_df.iterrows()]
    val_dataloader = PromptDataLoader(
        dataset = val_dataset,
        tokenizer = tokenizer,
        template = promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size = 32,
        shuffle=False
        )
    test_dataloader = PromptDataLoader(
        dataset = test_dataset,
        tokenizer = tokenizer,
        template = promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size = 32,
        shuffle=False
        )

    for hpo in all_param_combos:
        logging.info(f"Testing hyperparameter combination: {hpo}")
            # Data Loaders
        train_dataloader = PromptDataLoader(
            dataset = train_dataset,
            tokenizer = tokenizer,
            template = promptTemplate,
            tokenizer_wrapper_class=WrapperClass,
            batch_size = hpo['batch_size'],
            shuffle=False
        )
        
        # Prompt Model
        promptModel = PromptForClassification(
            template = promptTemplate,
            plm = plm,
            verbalizer = full_verbalizer
        )
        promptModel.to(DEVICE)
    
        # Optmiziers
        loss_fun = CrossEntropyLoss()
        plm_param = [
            {'params': [p for n, p in promptModel.plm.named_parameters()
                        if not any(nd in n for nd in ['bias', 'LayerNorm.weight'])],
             'weight_decay': 0.01},
            {'params': [p for n, p in promptModel.plm.named_parameters()
                        if any(nd in n for nd in ['bias', 'LayerNorm.weight'])],
             'weight_decay': 0.0}
        ]
        optimizers = [AdamW(plm_param, hpo['learning_rate'])]
    
        train()
    logging.info("Sweep Completed...")
    logging.info(f"Best model with hyperparameters {best_model['hpo']} and f1 on validation dataset: {best_model['F1']}")
    logging.info("Testing on test set...")
    promptModel = best_model['model']
    metrics = eval(test_dataloader, type='testing')
    logging.info(f"Evaluation on Test Set")
    print(metrics)
    logging.info(f'saving model at {args.best_model_path}')
    torch.save(promptModel.plm, args.best_model_path)
    logging.info("Script execution completed successfully.")


# === EXECUTION GUARD ===
if __name__ == "__main__":
    args = parse_arguments()
    main(args)