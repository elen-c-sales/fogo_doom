import sys
import time
import pygame
import imageio
import numpy as np

# ==========================================
# CONFIGURAÇÕES & CONSTANTES
# ==========================================

# Dimensões da simulação
LARGURA = 100
ALTURA = 100  # Altura vertical do fogo

# Configurações de Visualização
TAMANHO_JANELA = (800, 600)
FPS = 30
TEMPO_ANIMACAO = 3
TOTAL_FRAMES = FPS * TEMPO_ANIMACAO

# Paleta de Cores (Doom Fire)
CORES = [
    (7, 7, 7), (31, 7, 7), (47, 15, 7), (71, 15, 7), (87, 23, 7), (103, 31, 7), (119, 31, 7), (143, 39, 7),
    (159, 47, 7), (175, 63, 7), (191, 71, 7), (199, 71, 7), (223, 79, 7), (223, 87, 7), (223, 87, 7), (215, 95, 7),
    (215, 95, 7), (215, 103, 15), (207, 111, 15), (207, 119, 15), (207, 127, 15), (207, 135, 23), (199, 135, 23), (199, 143, 23),
    (199, 151, 31), (191, 159, 31), (191, 159, 31), (191, 167, 39), (191, 167, 39), (191, 175, 47), (183, 175, 47), (183, 183, 47),
    (183, 183, 55), (207, 207, 111), (223, 223, 159), (239, 239, 199), (255, 255, 255)
]

# Adiciona branco puro no final para garantir range correto e converte para uint8
PALETA = np.array(CORES + [(255, 255, 255)], dtype=np.uint8)


# ==========================================
# LÓGICA DE GERAÇÃO (DECAY MATRIX)
# ==========================================

def create_particle_streams(num_streams: int, max_age: int, width: int) -> np.ndarray:
    """
    Gera as 'biografias' das partículas.
    
    Imagine cada stream como uma linha horizontal de fogo que sobe.
    Calculamos toda a vida dessa linha, do nascimento (super quente) até morrer (frio).
    
    Args:
        num_streams: Quantas 'levas' de partículas precisamos
        max_age: Quanto tempo uma partícula vive (== ALTURA da tela, pois sobe 1px/frame)
        width: Largura da tela
        
    Returns:
        Array shape (num_streams, max_age, width) contendo valores de calor
    """
    
    BASE_HEAT = 36  # Calor máximo
    
    # 1. Gerar o decaimento aleatório para cada momento da vida
    # Cada partícula perde entre 0 e 2 pontos de calor por frame
    decay_amount = np.random.randint(0, 3, (num_streams, max_age, width))
    
    # 2. Somar o decaimento ao longo da vida
    # Ex: [1, 0, 2, 1] -> [1, 1, 3, 4] de perda acumulada
    # Aqui nem precisamos usar uma distribuição monotonicamente decrescente;
    # A operação de cumsum abstrai isso pra gente.
    total_decay_at_age = np.cumsum(decay_amount, axis=1)
    
    # 3. Calcular calor atual: Calor Inicial - Perda Acumulada
    # Usamos np.maximum para garantir que não fique negativo
    streams = np.maximum(0, BASE_HEAT - total_decay_at_age)
    
    return streams

def sample_space_time(streams: np.ndarray, total_frames: int, height: int, width: int) -> np.ndarray:
    """
    Descobre qual partícula existe em cada ponto do Espaço-Tempo.
    
    Para preencher um pixel na posição (Tempo T, Altura Y), precisamos responder:
    1. "Qual partícula está passando por aqui agora?" -> Identidade da Stream
    2. "Quão velha essa partícula é?" -> Idade
    """
    
    # 1. Criar grades de coordenadas para cada pixel do nosso "Cubo de Fogo"
    #    current_time_grid:  Matriz onde cada célula responde "Em que frame estou?"
    #    current_height_grid: Matriz onde cada célula responde "Em que altura estou?"
    
    # Broadcasting: T (tempo) varia no eixo 0, Y (altura) varia no eixo 1
    current_time_grid = np.arange(total_frames).reshape(-1, 1, 1)  # Shape: (Frames, 1, 1)
    current_height_grid = np.arange(height).reshape(1, -1, 1)      # Shape: (1, Altura, 1)
    
    # =========================================================
    # A FÍSICA DO MOVIMENTO VERTICAL CONSTANTE
    # =========================================================
    # Se todas as partículas sobem 1 pixel por frame:
    
    # PERGUNTA A: Qual a idade da partícula na altura Y?
    # RESPOSTA:   Exatamente Y frames.
    #             (Se subiu 50 pixels, viveu por 50 frames)
    particle_age_grid = current_height_grid
    
    # PERGUNTA B: Quando nasceu a partícula que está na altura Y no tempo T?
    # RESPOSTA:   Tempo Nascimento = Tempo Atual - Idade
    #             birth_time = T - Y
    particle_birth_time_grid = current_time_grid - particle_age_grid
    
    # =========================================================
    # CONVERSÃO PARA ÍNDICES DO ARRAY (LOOKUP)
    # =========================================================
    
    # Nossas streams são armazenadas num array onde o índice 0 é a primeira stream gerada.
    # Como simulamos streams "do passado" (tempo negativo) para preencher o início,
    # precisamos somar um offset para converter 'tempo de nascimento' em um índice válido.
    # A primeira stream (índice 0) nasceu em T = -height.
    offset_to_positive_index = height
    stream_index_grid = particle_birth_time_grid + offset_to_positive_index
    
    # Segurança: garantir que não acessamos índices fora do array
    max_valid_index = streams.shape[0] - 1
    stream_index_grid = np.clip(stream_index_grid, 0, max_valid_index)
    
    # AGORA O "CORTE":
    # Vamos buscar os valores de calor na nossa biblioteca de streams.
    #
    # streams[stream_id, age] -> retorna a linha inteira de largura W
    #
    # Precisamos remover a dimensão extra (=1) para usar como índices diretos
    idx_stream = stream_index_grid.squeeze(axis=-1)  # Shape: (Frames, Altura)
    idx_age = particle_age_grid.squeeze(axis=-1)     # Shape: (1, Altura) -> Broadcast para (Frames, Altura)
    
    # Busca vetorizada: "Para cada ponto (T,Y), pegue o calor da stream X na idade Z"
    fire_cube = streams[idx_stream, idx_age]
    
    return fire_cube

def apply_wind_effect(fire_cube: np.ndarray) -> np.ndarray:
    """
    Aplica efeito de vento deslocando pixels horizontalmente baseando-se na altura.
    Quanto mais alto (Y maior), mais deslocado para a esquerda.
    A lógica é a mesma da geração de partículas. Vamos gerar todo o vento de uma vez só em formato de matriz.
    """
    frames, height, width = fire_cube.shape
    
    # Criar grid de colunas originais
    # Shape: (1, 1, W)
    cols = np.arange(width).reshape(1, 1, -1)
    
    # Criar grid de linhas (altura) para calcular o vento
    # Shape: (1, H, 1)
    rows = np.arange(height).reshape(1, -1, 1)
    
    # Força do vento
    wind_strength = 0.5
    
    # Cálculo do Shift:
    # Altura 0 (base) -> shift 0
    # Altura 99 (topo) -> shift ~50
    shifts = (rows * wind_strength).astype(int)
    
    # Novas colunas = Coluna Original + Shift (com wrap-around usando %)
    # O vento sopra para a esquerda se somarmos, pois estamos rolando o índice
    new_cols_grid = (cols + shifts) % width
    
    # Aplicar o remapeamento usando fancy indexing
    # Para cada (t, y), usamos os novos índices de coluna calculados
    
    # Precisamos de índices explícitos para T e Y para parear com new_cols_grid
    # T: (F, 1, 1) broadcastable para (F, H, W)
    # Y: (1, H, 1) broadcastable para (F, H, W)
    T_idx = np.arange(frames).reshape(-1, 1, 1)
    Y_idx = np.arange(height).reshape(1, -1, 1)
    
    fire_with_wind = fire_cube[T_idx, Y_idx, new_cols_grid]
    
    return fire_with_wind

def gerar_fogo_matriz_decaimento():
    """Função principal que orquestra a geração."""
    
    # 1. Calcular quantas streams precisamos
    # Precisamos cobrir do passado (T - Height) até o futuro (Total Frames)
    # Range total = Total Frames + Altura
    num_streams = TOTAL_FRAMES + ALTURA
    
    # 2. Gerar as histórias de vida (Streams)
    streams = create_particle_streams(num_streams, ALTURA, LARGURA)
    
    # 3. Mapear para o Espaço-Tempo
    fire_cube = sample_space_time(streams, TOTAL_FRAMES, ALTURA, LARGURA)
    
    # 4. Aplicar Vento
    final_cube = apply_wind_effect(fire_cube)
    
    # IMPORTANTE:
    # Nossa lógica usou Y=0 como BASE e Y=99 como TOPO.
    # Mas bibliotecas gráficas (Pygame, ImageIO) usam Y=0 como TOPO.
    # Precisamos inverter o eixo Y para visualizar corretamente.
    final_cube_visual = final_cube[:, ::-1, :]
    
    return final_cube_visual

# ==========================================
# LOOP PRINCIPAL & VISUALIZAÇÃO
# ==========================================

def main():    
    # Geração Prévia
    start_time = time.time()
    fire_data = gerar_fogo_matriz_decaimento()
    end_time = time.time()

    print("\nConcluído!")
    print(f"Tempo de Geração: {(end_time - start_time)*1000:.2f}ms")
    print(f"Shape Final: {fire_data.shape} (Frames, Altura, Largura)")
    

    pygame.init()
    tela = pygame.display.set_mode(TAMANHO_JANELA)
    pygame.display.set_caption("Doom Fire - Decay Matrix (Refatorado)")
    clock = pygame.time.Clock()
    
    # Loop de Reprodução
    frames_gif = []
    frame_idx = 0
    rodando = True
    gif_saved = False
    
    while rodando:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                rodando = False
        
        # 1. Pegar o frame atual
        fogo_grid = fire_data[frame_idx]
        
        # 2. Colorir (Mapear índices de calor para RGB)
        rgb_array = PALETA[fogo_grid]
        
        # 3. Salvar frame para GIF (apenas na primeira passagem)
        if not gif_saved:
            if len(frames_gif) < TOTAL_FRAMES:
                frames_gif.append(rgb_array.copy())
            elif len(frames_gif) == TOTAL_FRAMES:
                print("\nSalvando GIF...")
                imageio.mimsave('fogo_decay_matrix.gif', frames_gif, fps=FPS, loop=0)
                print("GIF Salvo: fogo_decay_matrix.gif")
                gif_saved = True
        
        # 4. Renderizar no Pygame
        # Pygame espera (Largura, Altura, RGB), mas temos (Altura, Largura, RGB)
        # Precisamos transpor os eixos 0 e 1 (swapaxes)
        surface_array = np.swapaxes(rgb_array, 0, 1)
        
        surf = pygame.surfarray.make_surface(surface_array)
        pygame.transform.scale(surf, TAMANHO_JANELA, tela)
        pygame.display.flip()
        
        # Avançar frame
        frame_idx = (frame_idx + 1) % TOTAL_FRAMES
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
