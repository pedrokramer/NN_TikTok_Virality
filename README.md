<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Ilumlogo.pdf/page1-1200px-Ilumlogo.pdf.jpg" alt="logo_ilum" width="200"/>
<h1 align="center"> Viralidade de vídeos de TikTok! </h1>

## Bem-vindo(a)!

Esse repositório de GitHub foi desenvolvido na matéria de Redes Neurais e Algoritmos Genéticos no terceiro semestre da faculdade Ilum Escola de Ciência. Matéria essa ministrada pelo professor [Daniel Cassar](https://github.com/drcassar). O trabalho deste repositório é equivalente ao trabalho final da disciplina de Redes Neurais, e fora desenvolvido pelos alunos:
+ [Pedro Kramer](https://github.com/pedrokramer)
+ [Iasodara Lima](https://github.com/Iasodara)
+ [Daniel Bravin](https://github.com/MrBravin)

### Dados!

[tiktok_dataset.csv](https://github.com/pedrokramer/NN_TikTok_Virality/blob/main/tiktok_dataset.csv) são os dados que utilizamos, disponibilizados no [Kaggle](https://www.kaggle.com/datasets/yakhyojon/tiktok), de engajamento em vídeos do tiktok. Esse dataset contém 19383 exemplos(vídeos) e contém os seguintes atributos:
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
