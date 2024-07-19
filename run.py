from pathlib import Path

save_dir = '.'
save_dir = Path(save_dir) / 'classification_finetune'
save_dir.mkdir(parents=True, exist_ok=True)


save_dir = Path(save_dir) / 'pretrain'
save_dir.mkdir(parents=True, exist_ok=True)