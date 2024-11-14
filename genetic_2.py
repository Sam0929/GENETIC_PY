import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time

# Definindo parâmetros do algoritmo genético
POPULATION_SIZE = 200  # Escolhido para manter a diversidade sem aumentar muito o tempo de processamento
GENERATIONS = 1000  # Número máximo de gerações permitidas

# A taxa de mutação foi fixada em 10% para preservar diversidade no pool genético,
# ajudando a escapar de mínimos locais sem sacrificar a qualidade das soluções iniciais.
MUTATION_RATE = 0.1  # Taxa de mutação de 10% para garantir diversidade

CROSSOVER_RATE = 0.8  # Alta taxa de cruzamento para acelerar a convergência
ELITISM = 0.1  # 10% de elitismo para preservar as melhores soluções
STAGNATION_LIMIT = 50  # Parada após 50 gerações sem melhoria para evitar estagnação
DESIRED_DISTANCE = 0.05  # Distância alvo para uma solução aceitável, dependendo do problema

# --- Critério: Representação dos Pontos e do Gene ---
def generate_points(n, scenario='uniform'):
    if scenario == 'uniform':
        return [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
    elif scenario == 'circle':
        return [(50 + 40 * math.cos(2 * math.pi * i / n), 50 + 40 * math.sin(2 * math.pi * i / n)) for i in range(n)]

# --- Critério: Inicialização da População ---
def create_individual(points):
    individual = points[:]
    random.shuffle(individual)
    return individual

def create_population(points):
    return [create_individual(points) for _ in range(POPULATION_SIZE)]

# --- Critério: Função de Aptidão ---
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def fitness(individual):
    distance = 0
    for i in range(len(individual)):
        distance += calculate_distance(individual[i], individual[(i + 1) % len(individual)])
    return 1 / distance  # Inverso da distância para maximizar a aptidão

# --- Critério: Seleção, Cruzamento e Mutação ---
def selection(population, tournament_size=5):
    # Seleciona aleatoriamente um grupo de indivíduos para o torneio
    # Retorna o indivíduo com a maior aptidão no torneio
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=fitness)

def crossover(parent1, parent2):
    # Realiza um cruzamento baseado em partes de pai 1 e pai 2
    # Segmenta uma sequência aleatória do pai 1 e preenche o restante com a ordem do pai 2
    # para garantir que a solução resultante seja um caminho válido
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    child[start:end] = parent1[start:end]
    p2_indices = [p for p in parent2 if p not in child[start:end]]
    child = [p2_indices.pop(0) if x is None else x for x in child]
    return child

def mutate(individual):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]

# --- Critério: Avaliar e Mostrar o Desempenho ---
def evolve(population):
    new_population = sorted(population, key=fitness, reverse=True)[:int(POPULATION_SIZE * ELITISM)]
    while len(new_population) < POPULATION_SIZE:
        parent1 = selection(population)
        parent2 = selection(population)
        if random.random() < CROSSOVER_RATE:
            child = crossover(parent1, parent2)
        else:
            child = parent1[:]
        mutate(child)
        new_population.append(child)
    return new_population

# Função para salvar e exibir soluções de várias gerações

def plot_solutions(points, generations_solutions):
    num_solutions = len(generations_solutions)
    rows = (num_solutions - 1) // 5 + 1  # Calcular o número de linhas
    fig, axs = plt.subplots(rows, min(num_solutions, 5), figsize=(18, 10), dpi=80)
    fig.suptitle("Evolução das Soluções ao Longo das Gerações", fontsize=16)

    # Transformar axs em uma lista de listas, mesmo se for 1D, para garantir consistência
    if rows == 1:
        axs = np.array([axs]).reshape(1, -1) if num_solutions > 1 else np.array([[axs]])

    for i, (generation, individual) in enumerate(generations_solutions):
        ax = axs[i // 5, i % 5]
        x, y = zip(*individual + [individual[0]])
        ax.plot(x, y, marker='o')
        ax.set_title(f"Geração {generation}")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Função para registrar o progresso
def plot_progress(distances):
    plt.figure(figsize=(10, 5))
    plt.plot(distances)
    plt.xlabel('Geração')
    plt.ylabel('Melhor Distância')
    plt.title('Evolução da Melhor Distância ao Longo das Gerações')
    plt.show()

# --- Critério: Executar o Algoritmo e Mostrar Soluções ---
def genetic_algorithm(points, bonus=False):
    population = create_population(points)
    generations_solutions = []
    start_time = time.time()
    distances_progress = []
    distances_progress.append(float('inf'))
    best_distance = float('inf')
    stagnant_generations = 0

    for generation in range(GENERATIONS):
        population = evolve(population)
        best_individual = max(population, key=fitness)
        current_distance = 1 / fitness(best_individual)
        
        # Critério de parada por convergência (sem melhorias por várias gerações)
        if current_distance < best_distance:
            best_distance = current_distance
            stagnant_generations = 0
        else:
            stagnant_generations += 1
        
        # Mostrar a cada 50 gerações e a última geração
        if generation % 50 == 0 or generation == GENERATIONS - 1:
            print(f"Geração {generation}: Melhor distância = {best_distance:.2f}")
            generations_solutions.append((generation, best_individual))
        
        best_distance = 1 / fitness(best_individual)
        distances_progress.append(best_distance)  # Registra a melhor distância da geração

        # Critério de parada antecipada
        if stagnant_generations >= STAGNATION_LIMIT:
            print(f"Parando antecipadamente na geração {generation} por falta de melhorias.")
            break

        if generation > 0 and generation % 50 == 0:
            if 1 - (best_distance/distances_progress[generation-50]) <= DESIRED_DISTANCE:
                print(f"Parando antecipadamente na geração {generation}: distância ótima alcançada.")
                break

    end_time = time.time()
    if bonus:
        print(f"Tempo total de execução (critério bônus): {end_time - start_time:.2f} segundos")

    plot_solutions(points, generations_solutions)
    plot_progress(distances_progress)

# --- Critério: Teste com diferentes cenários ---
points_uniform = generate_points(8, 'uniform')
print("Cenário: Pontos Uniformes")
genetic_algorithm(points_uniform)

points_circle = generate_points(8, 'circle')
print("Cenário: Pontos em Círculo")
genetic_algorithm(points_circle)

# --- Critério Bônus: Teste com uma quantidade alta de pontos (modelo circular) ---
points_large_circle = generate_points(1000, 'circle')
print("\nCenário Bônus: Pontos em Círculo com Alta Quantidade de Pontos")
genetic_algorithm(points_large_circle, bonus=True)
