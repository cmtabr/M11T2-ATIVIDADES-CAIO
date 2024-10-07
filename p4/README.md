### Documentação do Código

Este código é uma implementação simplificada de um modelo de `Deep Learning` usando C++. Ele define estruturas para um tensor básico, uma camada densa e uma camada convolucional, e monta um modelo com essas camadas para realizar um "forward pass" de um tensor de entrada. Como requisitado na ponderada de número 4, a implementação é feita sem o uso de bibliotecas externas.

---

#### Classe `Tensor`

A classe `Tensor` serve como a estrutura de dados principal para armazenar e manipular matrizes multidimensionais (ou tensores). A implementação está focada em operações básicas como adição e multiplicação de matrizes.

- **Construtores:**
  - `Tensor()`: Construtor padrão que inicializa um tensor vazio.
  - `Tensor(const std::vector<std::vector<double>>& input_data)`: Construtor que inicializa o tensor com os dados fornecidos.

- **Métodos:**
  - `Tensor operator+(const Tensor& other) const`: Sobrecarga do operador `+` para somar dois tensores.
  - `Tensor dot(const Tensor& other) const`: Implementação da multiplicação matricial.
  - `std::vector<double> flatten() const`: Retorna uma versão achatada (1D) do tensor.
  - `void print() const`: Imprime o tensor no formato de matriz no console.
  - `std::pair<size_t, size_t> shape() const`: Retorna a forma do tensor (número de linhas e colunas).

---

#### Classe `Layer`

Esta é uma classe base abstrata para representar uma camada genérica. Qualquer classe que herde de `Layer` deve implementar o método `forward` para definir a operação que a camada realiza sobre o tensor de entrada.

- **Métodos:**
  - `virtual Tensor forward(const Tensor& input) = 0`: Método virtual puro que deve ser implementado pelas subclasses.

---

#### Classe `DenseLayer` (Herda de `Layer`)

Representa uma camada densa (ou camada totalmente conectada), onde cada neurônio está conectado a todos os neurônios da camada anterior.

- **Construtor:**
  - `DenseLayer(int input_size, int output_size)`: Inicializa a camada densa com pesos aleatórios e viés (bias) nulo. `input_size` é o número de neurônios de entrada, e `output_size` é o número de neurônios de saída.

- **Métodos:**
  - `Tensor forward(const Tensor& input) override`: Calcula a saída da camada densa aplicando a multiplicação matricial dos pesos e somando o bias.

---

#### Classe `Conv2D` (Herda de `Layer`)

Implementa uma camada convolucional 2D, usada geralmente para processamento de imagens. A camada convolucional aplica filtros sobre uma matriz de entrada, produzindo um mapa de ativação.

- **Construtor:**
  - `Conv2D(int num_filters, int kernel_size)`: Inicializa a camada convolucional com um número especificado de filtros, cada um com um tamanho de `kernel_size x kernel_size`. Os filtros são inicializados com valores aleatórios.

- **Métodos:**
  - `Tensor convolve(const Tensor& input)`: Aplica os filtros convolucionais sobre a entrada para produzir o mapa de ativação.
  - `Tensor forward(const Tensor& input) override`: Realiza a operação de convolução sobre o tensor de entrada.

---

#### Classe `Model`

Esta classe encapsula um modelo de rede neural composto por várias camadas (`Layer`). Fornece métodos para adicionar camadas e realizar um "forward pass" através de todas as camadas.

- **Métodos:**
  - `void add(Layer* layer)`: Adiciona uma nova camada ao modelo.
  - `Tensor forward(const Tensor& input)`: Executa um "forward pass" sobre todas as camadas do modelo.
  - `void summary()`: Imprime um resumo das camadas adicionadas ao modelo.
  - `~Model()`: Destrutor que libera a memória alocada para as camadas.

---

#### Função `main`

A função `main` implementa um exemplo de uso das classes definidas anteriormente:

1. Inicializa um tensor de entrada com uma matriz 5x5.
2. Cria um objeto `Model` e adiciona as seguintes camadas:
   - Uma camada `Conv2D` com 1 filtro e kernel de tamanho 5.
   - Uma camada `DenseLayer` com 9 entradas e 1 saída.
   - Outra camada `DenseLayer` com 1 entrada e 5 saídas.
3. Imprime o resumo do modelo.
4. Realiza o "forward pass" através do modelo e imprime a saída final.
