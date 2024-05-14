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
  + A matriz importante do SVD é a matriz U, que ao receber uma matriz BOW transposta (token X doc) devolve uma matriz token x 'token'. Entretanto, pela maneira que é organizado a matriz U do SVD funciona, as colunas perdem o significado de token ganhado dando valores de correlação dos tokens (linhas) com esse tópico indefinido.

### 


</p>
</details>
