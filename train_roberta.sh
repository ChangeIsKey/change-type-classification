

python3 src/roberta_trainer.py --train_path='datasets/train.jsonl' --dev_path='datasets/dev.jsonl'  --model_type "crossencoder" --pretrained_model "roberta-large" --evaluation "multi-task" --weight_decay 0.1 --output_path "models/" --lr 1e-6 --n_epochs 10

