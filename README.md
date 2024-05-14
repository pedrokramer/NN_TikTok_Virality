<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Ilumlogo.pdf/page1-1200px-Ilumlogo.pdf.jpg" alt="logo_ilum" width="200"/>
<h1 align="center"> Viralidade de vídeos de TikTok! </h1>

## Bem-vindo(a)!

Esse repositório de GitHub foi desenvolvido na matéria de Redes Neurais e Algoritmos Genéticos no terceiro semestre da faculdade Ilum Escola de Ciência. Matéria essa ministrada pelo professor [Daniel Cassar](https://github.com/drcassar). O trabalho deste repositório é equivalente ao trabalho final da disciplina de Redes Neurais, e fora desenvolvido pelos alunos:
+ [Pedro Kramer](https://github.com/pedrokramer) - 23013
+ [Iasodara Lima](https://github.com/Iasodara) - 23005
+ [Daniel Bravin](https://github.com/MrBravin) - 23020

<details>
    
<summary>Dados!</summary>
    
[tiktok_dataset.csv](https://github.com/pedrokramer/NN_TikTok_Virality/blob/main/tiktok_dataset.csv) são os dados que utilizamos, disponibilizados no [Kaggle](https://www.kaggle.com/datasets/yakhyojon/tiktok) , de engajamento em vídeos do tiktok. Esse dataset contém 19383 exemplos(vídeos) e contém os seguintes atributos: 
  
+ claim_status: Se o vídeo é um vídeo de afirmação ou opinião, é uma string que pode conter "claim" ou "opinion" como inputs.
+ video_id: É o número de identificação do vídeo.
+ video_duration_sec: É a duração do vídeo em segundos.
+ video_transcription_text: É a transcrição (Linguagem Natural) do que é falado nesse vídeo.
+ verified_status: É o status de verificação do perfil que postou o vídeo em questão. Uma string que pode conter "verified" ou "not verified".
+ author_ban_status: É o status de banimento do perfil que postou o vídeo. Pode conter três possíveis strings: “active”, “under scrutiny”, ou “banned”.
+ video_like_count: É a contagem de likes que o vídeo recebeu, em inteiro.
+ video_share_count: É a contagem de compartilhamentos que um vídeo recebeu. Em inteiro.
+ video_download_count: É a contagem de download que um vídeo recebeu. Em inteiro.
+ video_comment_count: É a contagem de comentários que um vídeo recebeu. Em inteiro.
+ video_view_count: Quantidade de visualizações. Nosso target!


</p>
</details>

<details>
    
<summary>Processamento de Linguagem Natural</summary>

  Todo o processo de tratamento de NL (Natural Language) foi feita no arquivo [LSA_COSINES.py](https://github.com/pedrokramer/NN_TikTok_Virality/blob/main/LSA_COSINES.py) . Os processos feitos foram os seguintes:
  
  ### Tokenização e Bag-of-Words
  + A tokenização foi feita pela biblioteca 'spacy'. Os tokens passaram por lemmarization: um processo que transforma palavras de mesma origem semântica em uma só, como transformar kissing, kissed, kiss em kiss. Além disso não foram pegadas pontuações ('!', '.', ',' e etc), espaços e stop-words (verbo to be, pronomes, "of", "from").
  + O processo de tokenização e criação da bag-of-words (BOW) foi simultâneo, a tokenização de cada exemplo criava uma nova chave para o dicionário "cont" que tinha como valor um dicionário que tinha chaves para cada palavra e quantidade que elas aparecem na setença daquele vídeo
  + Esse processo resulta numa matriz (doc x token) chamada de bag-of-words.
  + Foi tirado a palavra "moon" porque estava degenerada (tinha correlação com todos os tópicos de LSA, que será visto mais pra frente)
  + Foi tirado palavras que aparecem menos de 5 vezes que que aparecem mais de 1/1.1 ou aproximadamente 90% dos exemplos
    
  ### TF-iDF
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*V9ac4hLVyms79jl65Ym_Bw.jpeg" alt="tfidf" width="500"/>
  
  + O TF-iDF (term frequency-inverse document frequency) é uma medida estatística de o quão a palavra é relevante para um documento x num conjunto de documentos. Seu cálculo é de valor a valor da matriz BOW, e seu cálculo da pela imagem acima [Fonte](https://ted-mei.medium.com/demystify-tf-idf-in-indexing-and-ranking-5c3ae88c3fa0).
  + Foi feito pela biblioteca Scikit Learn [TfidfTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html).
  + Foi devolvido uma matriz BOW pós tratamento de TF-iDF (doc x token)
    
  ### Latent Semantic Analysis
  + Latent Semantic Analysis (LSA) ou Análise latente de semântica é um processo de criação de tópicos que tenta separar palavras em tópicos por meio de decomposição SVD da matriz transposta da BOW pós TF-iDF
  + A decomposição SVD usada foi o [Truncated SVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) do Scikit Learn, com o máximo de 100 composições
  + A matriz importante do SVD é a matriz U, que ao receber uma matriz BOW transposta (token X doc) devolve uma matriz token x 'token'. Entretanto, pela maneira que é organizado a matriz U do SVD funciona, as colunas perdem o significado de token ganhado dando valores de correlação dos tokens (linhas) com esse tópico indefinido. A matriz que representa a U (token x tópico) está salva como [U_SVD_tktk_.1.csv](https://github.com/pedrokramer/NN_TikTok_Virality/blob/main/U_SVD_tktk_.1.csv).
  + Pegando apenas as 10 palavras com maiores valores de cada tópico, foi feita uma matriz palavra(10) por tópico(100) chamada [Word_Vects_tktk_1.1.csv](https://github.com/pedrokramer/NN_TikTok_Virality/blob/main/Word_Vects_tktk_1.1.csv). Essa matriz é importante para ter uma ideia sobre o que cada tópico trata. Ela não é utilizada para outros fins além de análise visual.
  + Obs: Pela maneira que o SVD decompõe, os tópicos ficam em ordem de maior variância e menos degeneração de palavras, ou seja, a medida que os tópicos (colunas) vão passando, maior será a repetição de palavras entre os tópicos levando a tópicos sem variância entre si
  + Obs 2: Esses tópicos podem ser vistos como vetores num hiperplano onde cada token se torna uma base desse espaço

 ### Semelhança de Cossenos
Para fins de treino da Rede Neural, usar o valor de TF-iDF como um atributo de treinamento seria muito custoso, já que após a tokenização havia 568 tokens, e cada um deveria ser um dado de entrada. Então a maneira de solucionar isso foi tirando a semelhança de cosseno de cada descrição de vídeo, ou seja, cada linha da BOW com os 10 primeiros tópicos. (Essa escolha de tópicos foi arbritária, sob orientação do Prof. Amauri Jardim de Paula. Isso porque os tópicos, geralmente, após o 10º se tornam muito degenerados entre si). Os passos tomados então foram:
+ Pegar cada linha da matriz TF-iDF e calcular a semelhança de cosseno dos 10 primeiros tópicos da matriz transposta da matriz U (É necessário que seja a matriz U seja transposta para que ela fique em tópicos x token, para que as arrays que formam essa matriz tenham as mesmas dimensões as arrays da matriz TF-iDF [doc x token], assim a comparar linha por linha elas teram as mesmas dimensões)
+ Com todas as semelhanças de cosseno feitas entre as arrays de TF-iDF e 10ºs Tópicos, elas são normalizadas por max-mean e adicionadas no dataset com os outros atributos e targets chamado [tiktok_nlp_data.csv](https://github.com/pedrokramer/NN_TikTok_Virality/blob/main/tiktok_nlp_data.csv)


</p>
</details>

<details>
<summary>Tratamento dos Dados</summary>

Os seguintes tratamentos feitos no notebook [data_tiktok_treat.ipynb](https://github.com/pedrokramer/NN_TikTok_Virality/blob/main/data_tiktok_treat.ipynb) foram:

+ Os dados de 'claim_status' e 'verified_status' foram transformados em binário, 'claim' -> 1 e 'opinion' -> 0; 'verified' -> 1 e 'not verified' -> 0.
+ 'author_ban_status' passou por um processo de [OneHotEnconder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) do Scikit Learn.
+ 'video_duration_sec','video_view_count', 'video_like_count', 'video_share_count', 'video_download_count' e 'video_comment_count' tiveram seus valores normalizados pelo [MaxAbsScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html) do Scikit Learn
+ Todas alterações foram salvas em um arquivo csv chamado [tiktok_treated_data.csv](https://github.com/pedrokramer/NN_TikTok_Virality/blob/main/tiktok_treated_data.csv).

</p>
</details>

<details>
<summary>Rede Neural</summary>

A rede neural teve sua arquitetura e treinamentos feitos no notebook [Trabalho_REDES.ipynb](https://github.com/pedrokramer/NN_TikTok_Virality/blob/main/Trabalho_REDES.ipynb).

### Arquitetura da Rede
+ A rede é uma MLP (Multilayer Perceptron) e teve suas combinações de arquiteturas possíveis definidas na classe de python 'view_predictor_MLP'. Foi utilizado pytorch para esse processo
+ Tem função de perda de MSE
+ Foram definidas duas possíveis funções de ativação: ReLU ou Sigmoid (hiperparâmetro).
+ Tem o otimizador por descida do gradiente com taxa de aprendizado indefinida (hiperparâmetro).
+ Tem DropOut com taxa de dropout indefinida (hiperparâmetro).
+ Outros hiperparâmetros são a quantidade de camadas e de neurônios por camada.

  ### Treinamento Otimizado da Rede e busca de melhores hiperparâmetros
+ A backpropragation é feito com 1000 épocas. Foi usado a biblioteca Pytorch.
+ Para encontrar os melhores hiperparâmetros foi usado o Optuna. Foi feito um estudo de Optuna que fez 500 tentativas com MSE como métrica para minimizar, essa escolha por busca em grade
+ Melhor arquitetura encontrada explicitada no notebook [Trabalho_REDES.ipynb](https://github.com/pedrokramer/NN_TikTok_Virality/blob/main/Trabalho_REDES.ipynb).
</p>
</details>

