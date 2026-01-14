import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

SEED = 42 # Seed pour la reproductibilité des générations aléatoires
torch.manual_seed(SEED)

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval() # Met le modèle en mode évaluation (désactive dropout, etc.)

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")
max_gen_length = 50 # Longueur maximale pour toutes les générations

print(f"Seed utilisé: {SEED}")
print(f"Prompt: '{prompt}'")
print("-" * 50)


# Décodage Glouton
print("\n--- Décodage Glouton ---")
outputs_greedy = model.generate(
    **inputs,
    max_length=max_gen_length,
    do_sample=False,  # Désactive l'échantillonnage pour un choix déterministe
    num_beams=1,      # num_beams=1 avec do_sample=False équivaut au décodage glouton
    pad_token_id=tokenizer.eos_token_id # Évite les warnings lors de la génération en fin de séquence
)
text_greedy = tokenizer.decode(outputs_greedy[0], skip_special_tokens=True)
print("Génération Gloutonne:")
print(text_greedy)
print("-" * 50)


# Échantillonnage (Sampling)
print("\n--- Échantillonnage (Sampling) ---")
temperature_val = 0.7
top_k_val = 50
top_p_val = 0.95

def generate_once(seed_val, current_prompt_inputs, temp, top_k, top_p, penalty=1.0):
    torch.manual_seed(seed_val) # Fixe le seed pour cette génération spécifique
    out = model.generate(
        **current_prompt_inputs,
        max_length=max_gen_length,
        do_sample=True,      # Active l'échantillonnage
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=penalty,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

print(f"Paramètres d'échantillonnage: température={temperature_val}, top_k={top_k_val}, top_p={top_p_val}")
for s in [1, 2, 3, 4, 5]:
    generated_text = generate_once(s, inputs, temperature_val, top_k_val, top_p_val)
    print(f"Génération (SEED {s}):")
    print(generated_text)
    print("-" * 40)
print("-" * 50)


# Pénalité de répétition
print("\n--- Pénalité de répétition ---")
repetition_penalty_val = 2.0
seed_for_penalty = 1 # Utilise un seed fixe pour comparer

print(f"Génération SANS pénalité (Seed {seed_for_penalty}, paramètres par défaut):")
text_no_penalty = generate_once(seed_for_penalty, inputs, temperature_val, top_k_val, top_p_val, penalty=1.0)
print(text_no_penalty)

print(f"\nGénération AVEC pénalité (Seed {seed_for_penalty}, repetition_penalty={repetition_penalty_val}):")
text_with_penalty = generate_once(seed_for_penalty, inputs, temperature_val, top_k_val, top_p_val, penalty=repetition_penalty_val)
print(text_with_penalty)
print("-" * 50)


# Température très basse et très élevée
print("\n--- Température très basse et très élevée ---")
low_temp = 0.1
high_temp = 2.0
seed_for_temp = 1 # Un seed fixe pour comparer les températures

print(f"Génération (Température très BASSE={low_temp}, Seed {seed_for_temp}):")
text_low_temp = generate_once(seed_for_temp, inputs, low_temp, top_k_val, top_p_val)
print(text_low_temp)

print(f"\nGénération (Température très ÉLEVÉE={high_temp}, Seed {seed_for_temp}):")
text_high_temp = generate_once(seed_for_temp, inputs, high_temp, top_k_val, top_p_val)
print(text_high_temp)
print("-" * 50)


# Génération avec Beam Search
print("\n--- Génération avec Beam Search (num_beams=5) ---")
num_beams_val = 5

outputs_beam_5 = model.generate(
    **inputs,
    max_length=max_gen_length,
    num_beams=num_beams_val,
    early_stopping=True, # Arrête la génération une fois que tous les beams sont terminés
    do_sample=False,     # Le beam search est déterministe par nature, pas d'échantillonnage ici
    pad_token_id=tokenizer.eos_token_id
)
text_beam_5 = tokenizer.decode(outputs_beam_5[0], skip_special_tokens=True)
print(f"Génération Beam Search (num_beams={num_beams_val}):")
print(text_beam_5)
print("-" * 50)


# Performance du Beam Search
print("\n--- Performance du Beam Search (Mesure de temps) ---")
beam_sizes_to_test = [5, 10, 20] # Différentes valeurs de num_beams à tester
generation_times = {}

for beams in beam_sizes_to_test:
    start_time = time.time()
    model.generate(
        **inputs,
        max_length=max_gen_length,
        num_beams=beams,
        early_stopping=True,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    end_time = time.time()
    duration = end_time - start_time
    generation_times[beams] = duration
    print(f"Temps de génération pour num_beams={beams}: {duration:.4f} secondes")

print("\nRécapitulatif des temps:")
for beams, duration in generation_times.items():
    print(f"  num_beams={beams}: {duration:.4f}s")
print("-" * 50)