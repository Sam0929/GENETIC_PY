import random
import math
import numpy as np
import matplotlib.pyplot as plt

# --- Critério: Inicializar Parâmetros ---
# Definindo os parâmetros do algoritmo genético
POPULATION_SIZE = 100      # Tamanho da população
GENERATIONS = 500          # Quantidade de gerações
MUTATION_RATE = 0.1        # Taxa de mutação
CROSSOVER_RATE = 0.8       # Taxa de cruzamento
ELITISM = 0.1              # Porcentagem da população que será mantida por elitismo

# --- Critério: Representação dos Pontos e do Gene ---
# Gerar pontos uniformemente distribuídos ou em um círculo

def generate_points(n, scenario='uniform'):
    """Gera pontos em um plano cartesiano."""
    if scenario == 'uniform':
        return [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
    elif scenario == 'circle':
        return [(50 + 40 * math.cos(2 * math.pi * i / n), 50 + 40 * math.sin(2 * math.pi * i / n)) for i in range(n)]

# --- Critério: Inicialização da População ---
# Cada indivíduo é uma sequência (permutação) dos pontos
def create_individual(points):
    """Cria um indivíduo como uma permutação dos pontos."""
    individual = points[:]
    random.shuffle(individual)
    return individual

def create_population(points):
    """Cria uma população inicial aleatória."""
    return [create_individual(points) for _ in range(POPULATION_SIZE)]

# --- Critério: Função de Aptidão ---
def calculate_distance(point1, point2):
    """Calcula a distância euclidiana entre dois pontos."""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# --- Critério: Função de Aptidão ---
def fitness(individual):
    """Calcula a aptidão de um indivíduo (menor distância total) como o inverso da distância."""
    distance = 0
    for i in range(len(individual)):
        distance += calculate_distance(individual[i], individual[(i + 1) % len(individual)])
    return 1 / distance  # Inverso da distância para maximizar a aptidão


# --- Critério: Seleção, Cruzamento e Mutação ---
def selection(population):
    """Seleciona um par de indivíduos para cruzamento (seleção por torneio)."""
    return random.choices(population, k=2, weights=[fitness(ind) for ind in population])

def crossover(parent1, parent2):
    """Aplica crossover de ordem (Order Crossover - OX)"""
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    child[start:end] = parent1[start:end]
    p2_indices = [p for p in parent2 if p not in child[start:end]]
    child = [p2_indices.pop(0) if x is None else x for x in child]
    return child

def mutate(individual):
    """Realiza mutação trocando dois pontos de posição."""
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]

# --- Critério: Avaliar e Mostrar o Desempenho ---
def evolve(population):
    """Executa uma geração de evolução na população."""
    new_population = sorted(population, key=fitness, reverse=True)[:int(POPULATION_SIZE * ELITISM)]
    while len(new_population) < POPULATION_SIZE:
        parent1, parent2 = selection(population)
        if random.random() < CROSSOVER_RATE:
            child = crossover(parent1, parent2)
        else:
            child = parent1[:]
        mutate(child)
        new_population.append(child)
    return new_population

# Função para plotar a solução encontrada a cada época
def plot_solution(points, individual, epoch):
    """Plota a solução atual encontrada na geração especificada."""
    plt.figure()
    x, y = zip(*individual + [individual[0]])  # Fechar o ciclo
    plt.plot(x, y, marker='o')
    plt.title(f"Solução na Geração {epoch}")
    plt.show()

# --- Critério: Executar o Algoritmo e Mostrar Soluções ---
def genetic_algorithm(points):
    """Executa o algoritmo genético e exibe o desempenho a cada geração."""
    population = create_population(points)
    for generation in range(GENERATIONS):
        population = evolve(population)
        
        # Mostrar e acompanhar as soluções
        if generation % 50 == 0 or generation == GENERATIONS - 1:
            best_individual = max(population, key=fitness)
            print(f"Geração {generation}: Melhor distância = {-fitness(best_individual):.2f}")
            plot_solution(points, best_individual, generation)

# --- Critério: Teste com diferentes cenários ---
# Testar com pontos uniformes e pontos em círculo
points_uniform = generate_points(20, 'uniform')
points_circle = generate_points(20, 'circle')

print("Cenário: Pontos Uniformes")
genetic_algorithm(points_uniform)

print("Cenário: Pontos em Círculo")
genetic_algorithm(points_circle)
