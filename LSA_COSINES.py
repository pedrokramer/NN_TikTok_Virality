# Importações
import pandas as pd
import numpy as np
import math
import spacy
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

#Carregando o Data Frame
tiktok = 'tiktok_dataset.csv'
tiktok_data = pd.read_csv(tiktok)
tiktok_data = tiktok_data.dropna()
tiktok_data = tiktok_data.drop_duplicates(subset='video_transcription_text')

#Pegando Atributo em Linguagem natural
NL_data = tiktok_data.loc[:, 'video_transcription_text']
NL_data = NL_data.tolist()
nlp = spacy.load('en_core_web_sm') #Carregando o tokenizador do spacy

#Tokenizando e transformando em BOW
voca = [] #Vocabulário
cont = {} #Contagem das palavras por doc.
t0 = time.time()

for idx, doc in enumerate(NL_data):
    tokens = nlp(doc)
    filtered_tokens = [t.lemma_ for t in tokens if 
                      t.is_alpha and \
                      not t.is_punct and \
                      not t.is_space and \
                      not t.is_stop and \
                      t.pos_ in ['NOUN', 'VERB', 'ADJ']]
    cont[idx] = {} #Cria uma chave com o indice e com o valor de outro dic
    for token in filtered_tokens: 
        voca.append(token) #atribui 
        try:
            cont[idx][token] += 1
        except KeyError:
            cont[idx][token] = 1
voca_conj = set(voca)
tf = time.time()

print(f'{tf - t0} segundos')
print(len(voca))
print(len(voca_conj))


bow = pd.DataFrame(cont)
bow = bow.transpose()
bow = bow.fillna(0)

#Removendo palavras extremamente comuns ou muito raras

voca_to_remove = ['moon']#Palavra com correlação degenerada (observado empiricamente para esse conj de dados)

for token in list(voca_conj):
    freq = bow[token].sum()
    if freq < 5:
        voca_to_remove.append(token)
    elif freq > len(voca_conj)/1.1:
        voca_to_remove.append(token)

bow_2 = bow.drop(columns=voca_to_remove)
bow_2
bow_2 = bow.T.drop_duplicates().T #garantindo que não tem frases iguais apos tokenização

#transformação tf-idf

bow_matrix = bow_2.values

tfidf_transformer = TfidfTransformer()#atribuindo transf. a uma variável
tfidf_matrix = tfidf_transformer.fit_transform(bow_matrix)#fitando os valores da bow ao transf.tf-idf
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=bow_2.columns)#transformando em df

#preparação da bow para o SVD de sua matriz transposta

tfidf_df_ = tfidf_df.T.values #criando um array da transposta (transpostas estão indicadas com um '_' no final)

tfidf_df_ = tfidf_df_/tfidf_df_.max() #...
tfidf_df_ = tfidf_df_ - tfidf_df_.mean() #normalização max-mean dos valores da bow

#criando a matriz U da dec. SVD truncada

svd = TruncatedSVD(n_components=100) #com 100 componentes
U = svd.fit_transform(tfidf_df_) #fitando a bow tfidf transposta e retirando U

U_df = pd.DataFrame(U)
U_df.to_csv('U_SVD_tktk_.1.csv', index=False)

#criando vetores palavras utilizando a matriz U LSA

df_word_vecs = pd.DataFrame()

for n in range(100):
    top = U_df[n].tolist() #cria uma lista de um topico n
    top_sorted = top.copy() #cria uma cópia
    top_sorted.sort(reverse=True) #coloca do maior para o menor
    indices = [] #lista vazia
    
    for n_2 in range(10): #coloca os indices (da lista desordenada) das 10 palavras de maior valor (da lista ord)  
        n_2 = top_sorted[n_2] 
        idx = top.index(n_2)
        indices.append(idx)
        
    tokens_top = []
    for idx in indices: #adciona as 10 strings das palavras com maior valor de correlação achadas para os tópicos
        token = bow_2.columns[idx]
        tokens_top.append(token)
        
    #add uma coluna tópico com 10 linhas das palavras que vai da maior "corr" para menor 
    df_word_vecs[n] = tokens_top
    
# Word x Topic -> Topic x Word
df_word_vecs = df_word_vecs.transpose()
df_word_vecs.to_csv('Word_Vects_tktk_1.1.csv', index=False)

# cossenos dos 10 primeiros tópicos com os docs do df

U_t = U_df.values.T #para que as arrays que formam o df nas dimensões (topic x tokens)
tfidf_df = tfidf_df_.T #resgatando a bow (docs x tokens)
cosines = {} #cria um dic vazio

for ex in range(len(tfidf_df)): #iteração dos documentos
    cosines[ex] = {} #cria uma chave para cada doc com um dic de valor
    vector_ex = tfidf_df[ex] #cria uma var local do vetor documento
    
    for n in range(10):
        dot_product = np.dot(vector_ex, U_t[n]) #produto interno documento x topico
        norm_array1 = np.linalg.norm(vector_ex) #transforma o vetor num versor 
        norm_array2 = np.linalg.norm(U_t[n]) # ''                   ''
        cosine_similarity = dot_product / (norm_array1 * norm_array2) #faz a simi. de cosseno do documento com o topic n
        cosines[ex][f'cosine {n}'] = cosine_similarity #adiciona uma chave do cosine n com o valor da var acima
        
cosines_df = pd.DataFrame(cosines) # cos x doc
cosines_df = cosines_df.T # doc x cos

scaler = StandardScaler()
scaler.fit(cosines_df)
X_scaled = scaler.transform(cosines_df) #normalização padrão
cos_scaldf= pd.DataFrame(X_scaled, columns=['cosxt1','cosxt2','cosxt3','cosxt4','cosxt5','cosxt6','cosxt7','cosxt8','cosxt9','cosxt10',])

#Juntando em um csv
tktk_columns = tiktok_data[['claim_status', 'video_duration_sec', 'verified_status', 'author_ban_status', 'video_view_count', 'video_like_count', 'video_share_count', 'video_download_count', 'video_comment_count']]
tktk_columns_array = tktk_columns.values
cos_scal_array = cos_scaldf.values
array_f = np.concatenate((cos_scal_array, tktk_columns_array), axis=1)
df_final = pd.DataFrame(array_f, columns=['cosxt1','cosxt2','cosxt3','cosxt4','cosxt5','cosxt6','cosxt7','cosxt8','cosxt9','cosxt10','claim_status', 'video_duration_sec', 'verified_status', 'author_ban_status', 'video_view_count', 'video_like_count', 'video_share_count', 'video_download_count', 'video_comment_count'])

df_final.to_csv('tiktok_nlp_data.csv', index=False)