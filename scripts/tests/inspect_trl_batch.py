import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from trl import DPOTrainer, DPOConfig

device = torch.device("cpu")
model_name = "facebook/mbart-large-50"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.src_lang = "de_DE"
tokenizer.tgt_lang = "de_DE"

config = AutoConfig.from_pretrained(model_name)
config.encoder_layers = 2
config.decoder_layers = 2
config.d_model = 256
config.encoder_ffn_dim = 512
config.decoder_ffn_dim = 512
config.encoder_attention_heads = 4
config.decoder_attention_heads = 4
config.is_encoder_decoder = True

model = AutoModelForSeq2SeqLM.from_config(config).to(device)

sample_data = {
    "prompt": [
        "Der Bundestag hat heute nach langer Debatte ein neues Gesetz zur Förderung erneuerbarer Energien beschlossen.",
        "Die Bundesregierung plant umfassende Maßnahmen zur Bekämpfung des Klimawandels."
    ],
    "chosen": [
        "Das Parlament hat heute über ein neues Gesetz gesprochen. Das Gesetz soll mehr Öko-Strom fördern.",
        "Die Politik möchte mehr für den Natur-Schutz tun."
    ],
    "rejected": [
        "Der Bundestag beschloss heute debattiert ein erneuerbares Gesetz.",
        "Die Regierung plant Maßnahmen."
    ]
}
raw_dataset = Dataset.from_dict(sample_data)

dpo_config = DPOConfig(
    output_dir="results/tests/proof2_output",
    beta=0.1,
    max_length=128,
    per_device_train_batch_size=2,
    report_to="none",
    learning_rate=1e-5,
)

trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=raw_dataset,
    processing_class=tokenizer,
)

dataloader = trainer.get_train_dataloader()
batch = next(iter(dataloader))
print("Batch Keys:", batch.keys())
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"Key: {k}, Shape: {v.shape}, Dtype: {v.dtype}")

# Let's decode what is in input_ids
for i in range(len(batch["input_ids"])):
    print(f"\n--- Batch item {i} ---")
    print("Decoded input_ids:", tokenizer.decode(batch["input_ids"][i]))
    if "completion_mask" in batch:
        print("completion_mask sum:", batch["completion_mask"][i].sum().item())
        comp_ids = batch["input_ids"][i][batch["completion_mask"][i].bool()]
        print("Decoded completion:", tokenizer.decode(comp_ids))

try:
    loss = trainer.compute_loss(model, batch)
    print("\nLoss successfully computed:", loss.item())
except Exception as e:
    import traceback
    print("\nError in trainer.compute_loss:")
    traceback.print_exc()
