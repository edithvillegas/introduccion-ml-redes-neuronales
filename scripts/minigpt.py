"""
Un GPT diminuto a nivel de caracteres (inspirado en nanoGPT de Andrej Karpathy).

El mismo código aparece en el notebook 07; aquí se usa para precalcular la gráfica
de la diapositiva "Un GPT que aprende a escribir como Cervantes".
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoAtencion(nn.Module):
    """Atención causal con varias cabezas: cada token solo mira a los anteriores."""

    def __init__(self, d, n_cabezas, contexto, p_drop=0.1):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d)          # consultas, claves y valores en una sola matriz
        self.proy = nn.Linear(d, d)
        self.n_cabezas = n_cabezas
        self.drop = nn.Dropout(p_drop)
        self.register_buffer("mascara", torch.tril(torch.ones(contexto, contexto)).view(1, 1, contexto, contexto))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_cabezas, C // self.n_cabezas).transpose(1, 2)   # (B, cabezas, T, d_cabeza)
        k = k.view(B, T, self.n_cabezas, C // self.n_cabezas).transpose(1, 2)
        v = v.view(B, T, self.n_cabezas, C // self.n_cabezas).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))                   # similitudes QKᵀ/√d
        att = att.masked_fill(self.mascara[:, :, :T, :T] == 0, float("-inf"))    # no mirar al futuro
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)                  # promedio ponderado de valores
        return self.proy(y)


class Bloque(nn.Module):
    def __init__(self, d, n_cabezas, contexto):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.att = AutoAtencion(d, n_cabezas, contexto)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(0.1))

    def forward(self, x):
        x = x + self.att(self.ln1(x))   # flujo residual: cada capa suma su "aporte"
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab, d=128, n_capas=4, n_cabezas=4, contexto=128):
        super().__init__()
        self.contexto = contexto
        self.emb_tok = nn.Embedding(vocab, d)       # vector de cada carácter
        self.emb_pos = nn.Embedding(contexto, d)    # vector de cada posición
        self.bloques = nn.Sequential(*[Bloque(d, n_cabezas, contexto) for _ in range(n_capas)])
        self.ln = nn.LayerNorm(d)
        self.cabeza = nn.Linear(d, vocab)           # de vuelta a probabilidades sobre el vocabulario

    def forward(self, idx, objetivos=None):
        B, T = idx.shape
        x = self.emb_tok(idx) + self.emb_pos(torch.arange(T, device=idx.device))
        logits = self.cabeza(self.ln(self.bloques(x)))
        perdida = None
        if objetivos is not None:
            perdida = F.cross_entropy(logits.view(-1, logits.size(-1)), objetivos.view(-1))
        return logits, perdida


@torch.no_grad()
def generar(modelo, idx, n_nuevos, temperatura=1.0):
    """Genera n_nuevos tokens, uno a la vez, muestreando de la distribución predicha."""
    modelo.eval()
    for _ in range(n_nuevos):
        logits, _ = modelo(idx[:, -modelo.contexto:])
        probs = F.softmax(logits[:, -1, :] / temperatura, dim=-1)
        siguiente = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, siguiente], dim=1)
    return idx


def limpiar_gutenberg(texto):
    inicio = texto.index("En un lugar de la Mancha")
    fin = texto.index("*** END OF THE PROJECT GUTENBERG")
    return texto[inicio:fin].replace("\r\n", "\n")
