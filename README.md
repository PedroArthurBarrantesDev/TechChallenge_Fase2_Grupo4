|<p></p><p>FIAP – Pós Tech<br>Tech Challenger 02</p>|<p></p><p>Versão 1.0</p><p>Data de Criação</p><p>05/07/2024</p><p></p>|
| :-: | :- |


**Descrição do Problema**

A empresa Brisanet (https://www.brisanet.com.br), é uma empresa de 

telecomunicações, considerada a maior empresa do ramo na região nordeste do 

país.

No passado a empresa Brisanet, passou por um problema de alocação de técnicos 

para atendimento de OS (Ordem de serviço), nas cidades, por falta de controle, haviam vários técnicos atendendo a mesma OS, para instalação e também cancelamento de serviços.

O problema consistia em: 

\- Vários técnicos recebiam a mesma OS, tanto para instalação quanto para 

cancelamento de serviços. Fazendo com que o técnico se desloca-se para o 

endereço da OS, sem uma verificação da distância do técnico. Ou se já havia algum 

outro técnico atendendo aquela OS.

\- Os técnicos tinham que se comunicar por telefone, para saber se a OS aberta na 

central, estava sendo atendida por alguém. E por muitas vezes ocorria falha de 

comunicação, ou sobrecarregava algum técnico que ficava responsável pela 

organização das OS’s.


**

**PROPOSTA DE SOLUÇÃO**

Como solução, precisamos verificar os endereços das OS, e dividir entre os 

técnicos, montando uma rota de atendimento para o técnico, de acordo com sua 

localidade.

Cada OS não pode ser atendida por mais de 1 técnico, caso o técnico esteja 

disponível e já tenha atendido as suas OS’s, será disponibilizada um conjunto de  

OS que ainda não foram atendidas, para esse técnico, e a rota dessas OS’s devem 

ser definidas (algoritmo genético).  

Nesse caso apenas como é um MVP (Minimum Viable Product), não vamos considerar itens como o grau de urgência da OS e o tempo médio gasto em cada OS, o que determinaria a quantidade de OS’s atendidas diariamente por técnico.

**1.0 - Definindo as Rotas das OS’s**
**
`	`- **População inicial**: Após dividir as OS’s para cada técnico, 

vamos usar as OS’s e o técnico, para criar nossa **população inicial.**

\- **Gerações**: Eu proponho usarmos 100 gerações como valor default.

\- **Mutações**: Eu proponho usarmos o valor 0.5 de mutação como default.

\- **Fitness**: -.

**2.0 - Valores fictícios**

Podemos começar a desenvolver com um valor fictício de 10 técnicos e 30 

OS’s. Totalizando 3 OS’s para cada tecnico.

Cada Os com endereços diferentes.

|**Revisão**|**Data**|**Revisor/Criador**|<p>**Itens alterados**</p><p>(adicionar breve descrição da alteração)</p>|
| :-: | :-: | :-: | :-: |
|001|05/07/24|Fernando Meneses|Criação.|


