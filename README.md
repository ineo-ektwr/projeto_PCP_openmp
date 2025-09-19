# Projeto 1 de Prog Concorrente e Paralela

## Descrição
Sistemas de equações lineares, do tipo (A.x = b), podem ser resolvidos de
diversas formas. Uma delas  ́e por meio do método de Jacobi, que usa uma
abordagem iterativa, em que a cada itera ̧c ̃ao k se calcula uma nova aproximação
da solução.
A representaçãoo algébrica do m ́etodo de Jacobi  ́e dada pela composi ̧c ̃ao
entre uma matriz diagonal e as matrizes inferior e superior da matriz A. Usando
essa decomposição se pode calcular cada variável a partir da equação:

$$
x_{i}^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j \ne i} a_{ij} \, x_{j}^{(k)} \right)
$$


## O que fizemos?
Foi desenvolvido dois programas, sendo um deles uma versão sequencial para a solução do problema e o outro a versão paralela com openmp.


## Resultados
# Como executar (versao paralela)
- gcc -fopenmp -O2 openMP_Etore.c -lm -o jacobi_openmp 
- ./jacobi_openmp
