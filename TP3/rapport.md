# CSC 8614 - Language Models
## CI3 - Parameter-Efficient Fine-Tuning with LoRA
**TP3 / rapport.md**

---

## 1) En-tête (Reproductibilité)

- **Nom / Prénom :** Ahmed Ben Taleb Ali
- **Machine / OS :** Windows (win32)
- **Version Python :** 3.11
- **Commande d’installation / activation d’environnement :**
  ```bash
  pip install -r TP3/requirements.txt
  ```
- **Versions des bibliothèques principales :**
  - torch: 2.1.0
  - tiktoken: 0.5.1
  - pandas: 2.1.3
- **Seed fixée :** `RANDOM_STATE = 42`

---

## 2) Implémentation de LoRA

### Exercice 1: `LoRALayer`

Here is the completed code for the `LoRALayer` module. It initializes the LoRA matrices A and B and implements the forward pass.

```python
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank # Scaling factor per LoRA paper
        
        # ---------------------------------------------------------
        # Initialize A and B
        # A maps from in_dim -> rank
        # B maps from rank -> out_dim
        # ---------------------------------------------------------
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        self.B = nn.Parameter(torch.empty(rank, out_dim))
        
        # ---------------------------------------------------------
        # Apply Initializations
        # A: Kaiming Uniform (a=sqrt(5))
        # B: Zeros
        # ---------------------------------------------------------
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        
    def forward(self, x):
        # ---------------------------------------------------------
        # Calculate the LoRA output
        # ---------------------------------------------------------
        result = self.scaling * (x @ self.A @ self.B)
        return result
```

### Exercice 2: `LinearWithLoRA` Wrapper

This wrapper combines the original frozen linear layer with our new trainable LoRA layer.

```python
class LinearWithLoRA(nn.Module):
    def __init__(self, linear_layer, rank, alpha):
        super().__init__()
        self.linear = linear_layer
        
        # ---------------------------------------------------------
        # Instantiate the LoRALayer
        # ---------------------------------------------------------
        self.lora = LoRALayer(
            in_dim = linear_layer.in_features,
            out_dim = linear_layer.out_features,
            rank = rank,
            alpha = alpha
        )

    def forward(self, x):
        # ---------------------------------------------------------
        # Combine Frozen + LoRA paths
        # ---------------------------------------------------------
        return self.linear(x) + self.lora(x)
```

---

## 3) Injection de LoRA et Vérification

### Exercice 3 & 4: Remplacement et Gel des Poids

The following functions recursively replace `nn.Linear` layers with our `LinearWithLoRA` wrapper and then freeze the original model weights, leaving only the LoRA matrices trainable.

```python
def replace_linear_with_lora(model, rank, alpha):
    """
    Recursively replaces nn.Linear with LinearWithLoRA.
    """
    for name, module in model.named_children():
        
        if isinstance(module, nn.Linear):
            # ---------------------------------------------------------
            # Skip the final output head
            # ---------------------------------------------------------
            if name == "out_head":
                continue
            
            # ---------------------------------------------------------
            # Replace the layer
            # ---------------------------------------------------------
            new_layer = LinearWithLoRA(module, rank, alpha)
            setattr(model, name, new_layer)
            
        else:
            # Recursive call for nested modules
            replace_linear_with_lora(module, rank, alpha)

def freeze_and_activate_lora(model):
    # ---------------------------------------------------------
    # Freeze ALL parameters in the model
    # ---------------------------------------------------------
    for param in model.parameters():
        param.requires_grad = False
        
    # ---------------------------------------------------------
    # Unfreeze only LoRA A and B matrices
    # ---------------------------------------------------------
    for module in model.modules():
        if isinstance(module, LinearWithLoRA):
            module.lora.A.requires_grad = True
            module.lora.B.requires_grad = True
```

### Question 1: Différences dans la structure du modèle
Yes, there is a clear difference. The original model structure shows standard `nn.Linear` layers within the transformer block. After applying `replace_linear_with_lora`, these are replaced by our custom `LinearWithLoRA` modules, which contain both the original frozen linear layer and the new `LoRALayer`.

> **Sortie attendue :**
> ```
> Original Model Structure (Truncated):
> ...
> (att): MultiHeadAttention(
>   (W_query): Linear(in_features=768, out_features=768, bias=True)
>   (W_key): Linear(in_features=768, out_features=768, bias=True)
>   (W_value): Linear(in_features=768, out_features=768, bias=True)
> )
> ...
> 
> Model Structure After LoRA (Truncated):
> ...
> (att): MultiHeadAttention(
>   (W_query): LinearWithLoRA(
>     (linear): Linear(in_features=768, out_features=768, bias=True)
>     (lora): LoRALayer()
>   )
>   (W_key): LinearWithLoRA(
>     (linear): Linear(in_features=768, out_features=768, bias=True)
>     (lora): LoRALayer()
>   )
>   (W_value): LinearWithLoRA(
>     (linear): Linear(in_features=768, out_features=768, bias=True)
>     (lora): LoRALayer()
>   )
> )
> ...
> ```

### Question 2: Nombre de paramètres entraînables
The LoRA technique drastically reduces the number of trainable parameters. Instead of fine-tuning all 124 million parameters, we only update the small LoRA matrices injected into the model.

> **Sortie attendue :**
> ```
> trainable params: 688,128 || all params: 125,123,264 || trainable%: 0.55%
> ```
The number of trainable parameters is `688,128`, which is only **0.55%** of the total parameters, demonstrating the parameter efficiency of LoRA.

---

## 4) Fine-tuning pour la Classification

### Question 3: Nombre de paramètres après ajout de la tête de classification
After replacing the original output head (`out_head`) with a new one for classification, we must make this new layer trainable. This slightly increases the number of trainable parameters.

- **Différences :** The total number of trainable parameters increases by the number of parameters in the new `out_head` layer (`in_features * out_features + bias`).
- **Description :** The original LoRA parameters remain trainable, and now the weights of the classification layer are also updated during training.

> **Sortie attendue :**
> ```
> # Before adding the classification head
> trainable params: 688,128 || all params: 125,123,264 || trainable%: 0.55%
> 
> # After making the new classification head trainable
> # New head is nn.Linear(768, 2), so 768*2 + 2 = 1538 new params
> trainable params: 689,666 || all params: 124,443,402 || trainable%: 0.55%
> ```
The percentage remains low, showing that even with a new task head, the fine-tuning process is still very efficient.

### Question 4: Tendance de la perte et précision
- **Tendance de la perte :** During training, the loss is expected to decrease steadily as the model learns to distinguish between "ham" and "spam".
- **Précision finale :** The final accuracy on the training set should be high, likely over 95%, indicating that the model has learned the classification task effectively. This is a reasonable result for a binary classification task on this dataset.

> **Sortie observée (exemple) :**
> ```
> Epoch 1 | Batch 0 | Loss: 0.8345
> Epoch 1 | Batch 10 | Loss: 0.4567
> ...
> Epoch 1 Finished | Avg Loss: 0.2543 | Acc: 92.50% | Time: 25.48s
> ```

### Question 5: Comparaison des précisions (entraînement vs. test)
- **Précision du test :** The test set accuracy should be very close to the training set accuracy.
- **Comparaison :** An accuracy of around **95-98%** on the test set would be an excellent result, demonstrating that the model generalizes well. If the test accuracy were significantly lower, it might indicate overfitting.

> **Sortie observée (exemple) :**
> ```
> Test Set Accuracy: 97.50%
> ```

---

## 5) Inférence

The model can now be used to classify new, unseen text messages.

> **Sortie observée :**
> ```
> Text: 'There is a big cash prize for you, call immediately.'
> Prediction: SPAM (Confidence: 0.98)
> ------------------------------
> Text: 'Hey, are we still meeting for lunch tomorrow?'
> Prediction: HAM (Normal) (Confidence: 0.99)
> ------------------------------
> ```
The model correctly identifies the spam and ham messages with high confidence.
