import torch
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Charge le modèle pré-entraîné GPT-2 avec une tête de modélisation du langage.
# GPT2LMHeadModel ajoute une couche linéaire sur les sorties de GPT2Model pour prédire le token suivant.
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval() # Met le modèle en mode évaluation (désactive dropout, etc.)

phrase1 = "Artificial intelligence is fascinating."
inputs1 = tokenizer(phrase1, return_tensors="pt")
input_ids1 = inputs1["input_ids"][0]

with torch.no_grad():
    outputs1 = model(**inputs1)
    logits1 = outputs1.logits  # Les logits ont la forme (batch_size, seq_len, vocab_size)

# Convertir les logits en probabilités pour chaque token
probs1 = torch.softmax(logits1, dim=-1)

print(f"Probabilités conditionnelles pour la phrase: '{phrase1}'")
# Pour chaque token `t` (à partir du second), on affiche P(token_t | tokens_<t)
# La probabilité du token à la position `t` est donnée par les logits générés
# après avoir vu le token à la position `t-1`.
for t in range(1, len(input_ids1)):
    tok_id = input_ids1[t].item()
    p = probs1[0, t - 1, tok_id].item()
    tok_txt = tokenizer.decode([tok_id])
    print(f"  P( '{repr(tok_txt)}' | ... ) = {p:.3e}")
print("-" * 50)


def calculate_perplexity(phrase: str, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer):
    """Calcule la log-probabilité totale et la perplexité d'une phrase."""
    inputs = tokenizer(phrase, return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Utilise log_softmax pour des calculs numériquement stables avec les log-probabilités
    log_probs = torch.log_softmax(logits, dim=-1)
    
    total_logp = 0.0
    n = 0
    
    for t in range(1, len(input_ids)):
        tok_id = input_ids[t].item()
        # Récupère la log-probabilité du token `t` depuis la distribution prédite au temps `t-1`
        lp = log_probs[0, t - 1, tok_id].item()
        total_logp += lp
        n += 1

    if n == 0:
        return float('inf'), float('inf'), float('inf')

    avg_neg_logp = -total_logp / n
    ppl = math.exp(avg_neg_logp)
    
    return total_logp, avg_neg_logp, ppl

print("\nCalcul de log-probabilité et perplexité pour différentes phrases:")

# Phrase grammaticale
total_logp1, _, ppl1 = calculate_perplexity(phrase1, model, tokenizer)
print(f"Phrase: '{phrase1}'")
print(f"  Log-probabilité totale: {total_logp1:.2f}")
print(f"  Perplexité: {ppl1:.2f}\n")

# Phrase non grammaticale
phrase2 = "Artificial fascinating intelligence is."
total_logp2, _, ppl2 = calculate_perplexity(phrase2, model, tokenizer)
print(f"Phrase: '{phrase2}'")
print(f"  Log-probabilité totale: {total_logp2:.2f}")
print(f"  Perplexité: {ppl2:.2f}\n")

# Phrase en français
phrase3 = "L'intelligence artificielle est fascinante."
total_logp3, _, ppl3 = calculate_perplexity(phrase3, model, tokenizer)
print(f"Phrase: '{phrase3}'")
print(f"  Log-probabilité totale: {total_logp3:.2f}")
print(f"  Perplexité: {ppl3:.2f}")
print("-" * 50)


prefix = "Artificial intelligence is"
inp = tokenizer(prefix, return_tensors="pt")

with torch.no_grad():
    out = model(**inp)
    logits2 = out.logits

# Récupère les logits pour le *dernier* pas de temps de l'entrée,
# car c'est lui qui prédit le token suivant
last_logits = logits2[0, -1, :]
last_probs = torch.softmax(last_logits, dim=-1)

# Sélectionne les 10 tokens les plus probables
topk = 10
vals, idx = torch.topk(last_probs, k=topk)

print(f"\nTop {topk} prochains tokens après '{prefix}':")
for p, tid in zip(vals.tolist(), idx.tolist()):
    tok_str = tokenizer.decode([tid])
    print(f"  {repr(tok_str):<20} | Probabilité: {p:.3e}")
print("-" * 50)