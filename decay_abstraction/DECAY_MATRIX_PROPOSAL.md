# Proposta de Abstração por Matriz de Decaimento
## Um Estudo sobre Equivalência Perceptual em Simulação de Fluidos

---

### 1. Contexto e Problema

O estudo anterior (`TECHNICAL_DEEP_DIVE.md`) estabeleceu que o algoritmo de *Doom Fire* (Deschamps) possui uma limitação fundamental para paralelização total: a **dependência causal temporal**.

A equação de propagação:
$$Pixel[t] = f(Pixel[t-1], Random[t])$$

Exige que o estado $t-1$ seja completamente computado antes que o estado $t$ possa ser processado. Isso impõe um **loop sequencial** obrigatório, impedindo a vetorização total (O(1) loops).

### 2. A Proposta

A proposta do engenheiro [@victorvalentee](https://github.com/victorvalentee) sugere uma mudança de paradigma:

> **"Abstrair a condução do calor calculando uma 'matriz de decaimento' para cada 'pixel', onde o valor de decaimento é resultado de uma distribuição estatística monotonicamente decrescente."**

Em vez de *simular* a propagação física passo-a-passo, propomos **sintetizar** o resultado final baseando-se no comportamento estatístico esperado do fogo.

### 3. Fundamentação Teórica

Se analisarmos a "vida" de uma partícula de fogo que sobe da base, ela segue uma trajetória previsível estatisticamente:
1. Nasce com intensidade máxima (36) na base.
2. Sobe verticalmente (com deslocamentos horizontais pelo vento).
3. Perde intensidade (calor) a uma taxa variável.
4. Eventualmente se extingue (chega a 0).

Podemos modelar essa perda de intensidade não como uma subtração iterativa (`pixel = pixel - random`), mas como uma **curva de decaimento** pré-definida.

#### Abordagem Tradicional (Iterativa):
```python
intensidade = 36
for t in range(altura):
    decay = random.randint(0, 3)
    intensidade -= decay
    # Requer cálculo do passo anterior
```

#### Abordagem Proposta (Analítica/Estatística):
```python
# Pré-calcula o perfil de vida inteiro da partícula
vida_util = random.distribution(mean=15, std=5)
perfil = linear_decay(start=36, end=0, duration=vida_util)
# Não requer estado anterior, totalmente paralelizável
```

### 4. Hipótese de Pesquisa

**"A rigorosidade física da simulação de fluidos pode ser sacrificada em favor da velocidade, desde que o resultado visual seja indistinguível para o observador humano a 30 FPS."**

Estamos trocando **Equivalência Física** (o algoritmo original faz exatamente o que a física manda) por **Equivalência Perceptual** (parece fogo para quem vê).

### 5. Estratégia de Implementação

A nova implementação eliminará completamente o loop temporal, tratando o fogo como um tensor 3D estático onde o eixo Z é o tempo.

#### Pseudocódigo Vetorizado:

```python
# 1. Gerar Matriz de Decaimento (H x W x T)
# Cada coluna vertical representa a história de vida de um pixel
decay_profiles = np.random.exponential(scale=..., size=(H, W))

# 2. Construir o Volume 3D diretamente
# Aplicar broadcasting para gerar o volume sem iteração
fogo_volume = apply_decay_broadcast(base_heat=36, decay_profiles)

# 3. Aplicar Vento/Turbulência
# Shifts horizontais vetorizados (np.roll em massa)
fogo_volume = apply_wind_vectorized(fogo_volume)
```

### 6. Métricas de Sucesso

O experimento será considerado um sucesso se:

1. **Eliminação de Loops**: O código não contiver nenhum loop `for` explícito para geração dos frames.
2. **Performance**: O tempo de geração for significativamente menor que a abordagem atual (target: < 10ms para 90 frames).
3. **Qualidade Visual**: Em um teste cego lado-a-lado, a diferença entre a simulação real e a aproximada for difícil de notar.

---

### Próximos Passos

1. Implementar protótipo `fire_decay_matrix.py`
2. Criar visualização comparativa (Lado A: Deschamps / Lado B: Matriz de Decaimento)
3. Medir performance comparativa
