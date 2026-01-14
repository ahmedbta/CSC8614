from transformers import GPT2Tokenizer
import torch

# Fixe le seed pour la reproductibilité si des opérations aléatoires étaient présentes.
torch.manual_seed(42)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print(f"Taille du vocabulaire du tokenizer: {tokenizer.vocab_size}\n")

phrase = "Artificial intelligence is metamorphosing the world!"
print(f"Phrase originale: '{phrase}'")

# Tokenisation de la phrase
tokens = tokenizer.tokenize(phrase)
print("\nTokens:", tokens)

# Obtention des IDs correspondant aux tokens
# Note: tokenizer.encode() peut parfois différer de tokenizer.tokenize() + convert_tokens_to_ids()
# pour la gestion de tokens spéciaux. Pour un mapping clair, on utilise convert_tokens_to_ids.
ids_from_tokens = tokenizer.convert_tokens_to_ids(tokens)
print("Token IDs:", ids_from_tokens)

print("\nDétails par token (Token -> ID -> Décodage):")
for i in range(len(tokens)):
    token_str = tokens[i]
    token_id = ids_from_tokens[i]
    decoded_txt = tokenizer.decode([token_id])
    print(f"- ID: {token_id:<5} | Token: {repr(token_str):<20} | Décodé: {repr(decoded_txt)}")
print("-" * 50)

phrase2 = "GPT models use BPE tokenization to process unusual words like antidisestablishmentarianism."
print(f"\nPhrase à expérimenter: '{phrase2}'")

tokens2 = tokenizer.tokenize(phrase2)
print("\nTokens pour la phrase expérimentale:", tokens2)

# Logique pour isoler et compter les sous-tokens pour "antidisestablishmentarianism"
long_word = "antidisestablishmentarianism"
sub_tokens_for_long_word = []
reconstructed_word = ""
found = False

for token in tokens2:
    clean_token = token.replace('Ġ', '')
    
    # Détecte le début du word long dans les tokens
    if not found and long_word.startswith(clean_token):
        found = True
    
    if found:
        sub_tokens_for_long_word.append(token)
        reconstructed_word += clean_token
        # Arrête la reconstruction une fois le word complet retrouvé
        if reconstructed_word == long_word:
            break
        # Si la reconstruction dévie, réinitialise (sécurité)
        elif not long_word.startswith(reconstructed_word):
            sub_tokens_for_long_word = [] 
            break

print(f"\nLe mot '{long_word}' est découpé en {len(sub_tokens_for_long_word)} sous-tokens:")
print(sub_tokens_for_long_word)
print("-" * 50)