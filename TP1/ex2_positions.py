import torch
from transformers import GPT2Model
import plotly.express as px
from sklearn.decomposition import PCA

# Charger le modèle pré-entraîné GPT-2
model = GPT2Model.from_pretrained("gpt2")

# Récupérer la matrice des embeddings positionnels (wpe pour Word Position Embeddings).
# .weight contient les poids appris du module d'embedding.
position_embeddings = model.wpe.weight

print(f"Shape des embeddings positionnels: {position_embeddings.size()}")
print(f"Dimension d'embedding (n_embd): {model.config.n_embd}")
print(f"Taille de contexte max (n_positions): {model.config.n_positions}")
print("-" * 50)


# Visualisation PCA pour les 50 premières positions
# Extraire les 50 premières positions et les convertir en numpy.
# .detach() est nécessaire pour détacher le tenseur du graphe de calcul.
# .cpu() déplace le tenseur vers le CPU si nécessaire.
positions_50 = position_embeddings[:50].detach().cpu().numpy()

# Réduction de dimension avec PCA
pca_50 = PCA(n_components=2)
reduced_50 = pca_50.fit_transform(positions_50)

# Création du graphique interactif
fig_50 = px.scatter(
    x=reduced_50[:, 0],
    y=reduced_50[:, 1],
    text=[str(i) for i in range(len(reduced_50))],
    color=list(range(len(reduced_50))),
    title="Encodages positionnels de GPT-2 (PCA, positions 0-49)",
    labels={"x": "Première composante principale (PCA 1)", "y": "Deuxième composante principale (PCA 2)"}
)
fig_50.update_traces(textposition='top center')

path_50 = "TP1/positions_50.html"
fig_50.write_html(path_50)
print(f"Graphique sauvegardé dans : {path_50}")
print("-" * 50)


# Visualisation PCA pour les 200 premières positions
positions_200 = position_embeddings[:200].detach().cpu().numpy()

# Réduction de dimension
pca_200 = PCA(n_components=2)
reduced_200 = pca_200.fit_transform(positions_200)

# Création du graphique
fig_200 = px.scatter(
    x=reduced_200[:, 0],
    y=reduced_200[:, 1],
    text=[str(i) for i in range(len(reduced_200))],
    color=list(range(len(reduced_200))),
    title="Encodages positionnels de GPT-2 (PCA, positions 0-199)",
    labels={"x": "Première composante principale (PCA 1)", "y": "Deuxième composante principale (PCA 2)"}
)
fig_200.update_traces(textposition='top center')

path_200 = "TP1/positions_200.html"
fig_200.write_html(path_200)
print(f"Graphique sauvegardé dans : {path_200}")
print("-" * 50)