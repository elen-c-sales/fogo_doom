import pygame
import numpy as np
import sys

# Configuracoes
LARGURA_FOGO = 100
ALTURA_FOGO = 100
TAMANHO_JANELA = (800, 600)
FPS = 30

# Paleta do Doom (RGB)
CORES = [
    (7, 7, 7), (31, 7, 7), (47, 15, 7), (71, 15, 7), (87, 23, 7), (103, 31, 7), (119, 31, 7), (143, 39, 7),
    (159, 47, 7), (175, 63, 7), (191, 71, 7), (199, 71, 7), (223, 79, 7), (223, 87, 7), (223, 87, 7), (215, 95, 7),
    (215, 95, 7), (215, 103, 15), (207, 111, 15), (207, 119, 15), (207, 127, 15), (207, 135, 23), (199, 135, 23), (199, 143, 23),
    (199, 151, 31), (191, 159, 31), (191, 159, 31), (191, 167, 39), (191, 167, 39), (191, 175, 47), (183, 175, 47), (183, 183, 47),
    (183, 183, 55), (207, 207, 111), (223, 223, 159), (239, 239, 199), (255, 255, 255)
]
PALETA = np.array(CORES + [(255, 255, 255)], dtype=np.uint8)

def atualizar_fogo_deschamps(fogo_grid):
    decay = np.random.randint(0, 3, fogo_grid.shape)
    src_view = fogo_grid[1:, :]
    
    decay_sliced = decay[1:, :]
    decay_heat = decay_sliced
    novo_calor = np.maximum(0, src_view - decay_heat)
    
    fogo_grid[:-1, :] = novo_calor
    # Vento constante para a esquerda
    fogo_grid[:-1, :] = np.roll(fogo_grid[:-1, :], -1, axis=1) 
    
    # Aleatoriedade horizontal linha a linha
    rows, cols = fogo_grid.shape
    shifts = np.random.randint(-1, 2, rows)
    for i in range(rows - 1):
        fogo_grid[i, :] = np.roll(fogo_grid[i, :], shifts[i])

    return fogo_grid

def main():
    pygame.init()
    tela = pygame.display.set_mode(TAMANHO_JANELA)
    pygame.display.set_caption("Doom Fire - Elen Camila Sales")
    clock = pygame.time.Clock()
    
    fogo_grid = np.zeros((ALTURA_FOGO, LARGURA_FOGO), dtype=np.int32)
    fogo_grid[-1, :] = 36 

    rodando = True
    while rodando:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                rodando = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                fogo_grid[:-1, :] = 0

        fogo_grid = atualizar_fogo_deschamps(fogo_grid)

        rgb_array = PALETA[fogo_grid]
        surface_array = np.swapaxes(rgb_array, 0, 1)
        surf = pygame.surfarray.make_surface(surface_array)
        
        pygame.transform.scale(surf, TAMANHO_JANELA, tela)
        
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()