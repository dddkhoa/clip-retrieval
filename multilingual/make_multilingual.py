import gzip
import logging
import os
import sys
from datetime import datetime

import numpy as np
import open_clip
import torch
import wandb
from sentence_transformers import (
    LoggingHandler,
    SentenceTransformer,
    evaluation,
    losses,
    models,
)
from sentence_transformers.datasets import ParallelSentencesDataset
from torch.utils.data import DataLoader

from model_wrapper import OpenClipWrapper

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)

teacher_model_name = "ryanyip7777/pmc_vit_l_14"  # Our monolingual teacher model, we want to convert to multiple languages
student_model_name = "kaiserrr/Bilingual-BioSimCSE-BioLinkBERT-base"  # Multilingual base model we use to imitate the teacher model

max_seq_length = 128  # Student model max. lengths for inputs (number of word pieces)
train_batch_size = 64  # Batch size for training
inference_batch_size = 64  # Batch size at inference
max_sentences_per_trainfile = (
    500000  # Maximum number of  parallel sentences for training
)
train_max_sentence_length = (
    250  # Maximum length (characters) for parallel training sentences
)

num_epochs = 10  # Train for x epochs
num_warmup_steps = 10000  # Warumup steps

num_evaluation_steps = 1000  # Evaluate performance after every xxxx steps

output_path = "output/make-multilingual-sys-" + datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S"
)

train_files = []
dev_files = []
is_dev_file = False
for arg in sys.argv[1:]:
    if arg.lower() == "--dev":
        is_dev_file = True
    else:
        if not os.path.exists(arg):
            print("File could not be found:", arg)
            exit()

        if is_dev_file:
            dev_files.append(arg)
        else:
            train_files.append(arg)

if len(train_files) == 0:
    print("Please pass at least some train files")
    print(
        "python make_multilingual_sys.py file1.tsv.gz file2.tsv.gz --dev dev1.tsv.gz dev2.tsv.gz"
    )
    exit()

logger.info("Train files: {}".format(", ".join(train_files)))
logger.info("Dev files: {}".format(", ".join(dev_files)))

logger.info("Load teacher model")

teacher_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:ryanyip7777/pmc_vit_l_14')
teacher_model = OpenClipWrapper(teacher_model, device="cuda:0" if torch.cuda.is_available() else "cpu")


logger.info("Create student model from scratch")
word_embedding_model = models.Transformer(
    student_model_name, max_seq_length=max_seq_length, tokenizer_args={"report_to": "wandb"}
)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
student_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Set up wandb for monitoring training process
wandb.init(
    project="semantic-search",
    
    # track hyperparameters and run metadata
    config = {
    "lr": 2e-5,
    "eps": 1e-6,
    "epochs": num_epochs,
    "max_seq_length": max_seq_length,  
    "train_batch_size": train_batch_size,  
    "inference_batch_size": inference_batch_size,  
    "max_sentences_per_trainfile": max_sentences_per_trainfile,
    "train_max_sentence_length": train_max_sentence_length,
    "num_warmup_steps": num_warmup_steps  
    }
)
wandb.watch(student_model, log_freq=num_evaluation_steps)

train_data = ParallelSentencesDataset(
    student_model=student_model,
    teacher_model=teacher_model,
    batch_size=inference_batch_size,
    use_embedding_cache=True,
)
for train_file in train_files:
    train_data.load_data(
        train_file,
        max_sentences=max_sentences_per_trainfile,
        max_sentence_length=train_max_sentence_length,
    )

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MSELoss(model=student_model)

# Evaluate cross-lingual performance on different tasks
evaluators = (
    []
)  # evaluators has a list of different evaluator classes we call periodically

for dev_file in dev_files:
    logger.info("Create evaluator for " + dev_file)
    src_sentences = []
    trg_sentences = []
    with gzip.open(dev_file, "rt", encoding="utf8") if dev_file.endswith(
        ".gz"
    ) else open(dev_file, encoding="utf8") as fIn:
        for line in fIn:
            splits = line.strip().split("\t")
            if splits[0] != "" and splits[1] != "":
                src_sentences.append(splits[0])
                trg_sentences.append(splits[1])

    # Mean Squared Error (MSE) measures the (euclidean) distance between teacher and student embeddings
    dev_mse = evaluation.MSEEvaluator(
        src_sentences,
        trg_sentences,
        name=os.path.basename(dev_file),
        teacher_model=teacher_model,
        batch_size=inference_batch_size,
    )
    evaluators.append(dev_mse)

    # TranslationEvaluator computes the embeddings for all parallel sentences. It then check if the embedding of
    # source[i] is the closest to target[i] out of all available target sentences
    # dev_trans_acc = evaluation.TranslationEvaluator(
    #     src_sentences,
    #     trg_sentences,
    #     name=os.path.basename(dev_file),
    #     batch_size=inference_batch_size,
    # )
    # evaluators.append(dev_trans_acc)



def callback(score, epoch, step):
    wandb.log({"score": score})  # score = -MSE 


student_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluation.SequentialEvaluator(
        evaluators, main_score_function=lambda scores: np.mean(scores)
    ),
    epochs=num_epochs,
    warmup_steps=num_warmup_steps,
    evaluation_steps=num_evaluation_steps,
    output_path=output_path,
    save_best_model=True,
    optimizer_params={"lr": 2e-5, "eps": 1e-6},
    callback=callback
)

wandb.finish()
