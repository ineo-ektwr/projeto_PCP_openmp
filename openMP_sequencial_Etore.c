#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// --- CONSTANTES ---
#define MAX_ITERACOES 10000
#define N 10
// Critério de parada
#define TOLERANCIA 1e-5
// Nomes dos arquivos
#define ARQUIVO_ENTRADA "linear10.dat"
#define ARQUIVO_SAIDA "saida10.dat"

// --- PROTÓTIPOS DAS FUNÇÕES ---

void ler_dados(double **A, double *b);
void escrever_saida(double *x);
void liberar_memoria(double **A, double *b, double *x_atual, double *x_proximo);
void verificar_dominancia_diagonal(double **A);

// --- FUNÇÃO PRINCIPAL ---
int main()
{

    // --- 1. Alocação de Memória ---
    double **A = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++)
    {
        A[i] = (double *)malloc(N * sizeof(double));
    }
    double *b = (double *)malloc(N * sizeof(double));
    double *x_atual = (double *)malloc(N * sizeof(double));
    double *x_proximo = (double *)malloc(N * sizeof(double));

    printf("Memoria alocada para o sistema %dx%d.\n", N, N);

    // --- 2. Leitura dos Dados de Entrada ---
    ler_dados(A, b);
    printf("Dados lidos do arquivo '%s'.\n", ARQUIVO_ENTRADA);

    verificar_dominancia_diagonal(A);

    // --- 3. Inicialização ---
    // Começamos com uma solução inicial de zeros
    for (int i = 0; i < N; i++)
    {
        x_atual[i] = 0.0;
    }

    // --- 4. Método de Jacobi (Lógica Principal) ---
    int iteracoes = 0;
    double max_diff; // Maior diferença entre x_proximo e x_atual

    printf("Iniciando o metodo de Jacobi...\n");
    do
    {
        // Calcula o vetor da próxima iteração (x_proximo)
        for (int i = 0; i < N; i++)
        {
            double soma = 0.0;
            for (int j = 0; j < N; j++)
            {
                if (i != j)
                {
                    soma += A[i][j] * x_atual[j];
                }
            }
            x_proximo[i] = (b[i] - soma) / A[i][i];
        }

        // Verifica o critério de parada (maior diferença)
        max_diff = 0.0;
        for (int i = 0; i < N; i++)
        {
            double diff = fabs(x_proximo[i] - x_atual[i]);
            if (diff > max_diff)
            {
                max_diff = diff;
            }
        }

        // Atualiza o vetor da solução para a próxima iteração
        for (int i = 0; i < N; i++)
        {
            x_atual[i] = x_proximo[i];
        }

        iteracoes++;
        printf("Iteracao %d, max_diff = %e\n", iteracoes, max_diff);

        if (iteracoes > MAX_ITERACOES)
        {
            printf("O método não convergiu após %d iterações.\n", MAX_ITERACOES);
            break; // Sai do loop do-while
        }

    } while (max_diff >= TOLERANCIA);

    printf("Convergencia alcancada em %d iteracoes.\n", iteracoes);

    // --- 5. Escrita do Resultado ---
    escrever_saida(x_atual);
    printf("Resultado salvo em '%s'.\n", ARQUIVO_SAIDA);

    // --- 6. Liberação de Memória ---
    liberar_memoria(A, b, x_atual, x_proximo);
    printf("Memoria liberada com sucesso.\n");

    return 0;
}

// --- IMPLEMENTAÇÃO DAS FUNÇÕES ---

/**
 * @brief Lê os dados da matriz A e do vetor b a partir do arquivo de entrada.
 */
void ler_dados(double **A, double *b)
{
    FILE *arquivo = fopen(ARQUIVO_ENTRADA, "r");
    if (arquivo == NULL)
    {
        perror("Erro ao abrir o arquivo de entrada");
        exit(EXIT_FAILURE);
    }

    // Leitura da matriz A
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            fscanf(arquivo, "%lf", &A[i][j]);
        }
    }

    // Leitura do vetor b
    for (int i = 0; i < N; i++)
    {
        fscanf(arquivo, "%lf", &b[i]);
    }

    fclose(arquivo);
}

/**
 * @brief Escreve o vetor solução x no arquivo de saída.
 */
void escrever_saida(double *x)
{
    FILE *arquivo = fopen(ARQUIVO_SAIDA, "w");
    if (arquivo == NULL)
    {
        perror("Erro ao abrir o arquivo de saida");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < N; i++)
    {
        // Escreve com 4 casas decimais e um espaço
        fprintf(arquivo, "%10.4f ", x[i]);
    }
    // fprintf(arquivo, "\n"); 

    fclose(arquivo);
}

/**
 * @brief Libera toda a memória alocada dinamicamente.
 */
void liberar_memoria(double **A, double *b, double *x_atual, double *x_proximo)
{
    for (int i = 0; i < N; i++)
    {
        free(A[i]);
    }
    free(A);
    free(b);
    free(x_atual);
    free(x_proximo);
}

void verificar_dominancia_diagonal(double **A) {
    printf("\n--- Verificando Dominancia Diagonal ---\n");
    int dominante = 1; // Assumimos que é dominante até provar o contrário

    for (int i = 0; i < N; i++) {
        double soma_off_diagonal = 0.0;
        for (int j = 0; j < N; j++) {
            if (i != j) {
                soma_off_diagonal += fabs(A[i][j]);
            }
        }

        double diagonal = fabs(A[i][i]);
        printf("Linha %2d: |Diagonal| = %9.4f, Soma |Resto| = %9.4f. ", i, diagonal, soma_off_diagonal);

        if (diagonal <= soma_off_diagonal) {
            printf("--> NAO E DOMINANTE!\n");
            dominante = 0;
        } else {
            printf("--> OK.\n");
        }
    }

    if (dominante) {
        printf("A matriz e diagonalmente dominante. O metodo deve convergir.\n");
    } else {
        char opc;
        printf("ATENCAO: A matriz NAO e diagonalmente dominante. O metodo de Jacobi pode divergir.\n");
        printf("\nContinuar? (y/n) ");
        scanf("%c", &opc);

        if (opc != 'y' && opc != 'Y')
            exit(EXIT_SUCCESS);
    }
    printf("---------------------------------------\n\n");
}
