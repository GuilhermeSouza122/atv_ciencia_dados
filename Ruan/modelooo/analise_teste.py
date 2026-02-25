import pandas as pd
import numpy as np
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

CLIENT_ID = '5cfa34daba304d2fa09b386d91e2d786'
CLIENT_SECRET = 'b77e1d9e4cc445988cae9a1abfa8166e'

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))

caminho_json = r'C:\Users\ruanlourenco-ieg\OneDrive - Instituto Germinare\Área de Trabalho\3ano\Ciencia de Dados\atv_ciencia_dados\Ruan\YourSoundCapsule.json'
caminho_csv = r'C:\Users\ruanlourenco-ieg\OneDrive - Instituto Germinare\Área de Trabalho\3ano\Ciencia de Dados\atv_ciencia_dados\Ruan\dataset.csv'

with open(caminho_json, encoding='utf-8') as f:
    capsule_data = json.load(f)

historico = []
for stat in capsule_data['stats']:
    for track in stat.get('topTracks', []):
        historico.append({
            'track_name': track['name'],
            'streamCount': track['streamCount']
        })

df_historico = pd.DataFrame(historico)
df_historico = df_historico.groupby('track_name', as_index=False).sum()

df_top_historico = df_historico.sort_values(by='streamCount', ascending=False).head(500).copy()

def obter_spotify_id(track_name):
    try:
        resultados = sp.search(q=track_name, type='track', limit=1)
        if resultados['tracks']['items']:
            return resultados['tracks']['items'][0]['id']
    except Exception as e:
        pass
    return None

df_top_historico['track_id'] = df_top_historico['track_name'].apply(obter_spotify_id)
df_top_historico = df_top_historico.dropna(subset=['track_id'])

df_kaggle = pd.read_csv(caminho_csv)
df_kaggle = df_kaggle.drop_duplicates(subset=['track_id'])

df_final = pd.merge(df_top_historico, df_kaggle, on='track_id', how='inner')

df_final = df_final.rename(columns={'track_name_x': 'track_name'})


if len(df_final) > 10:
    df_model = df_final[(df_final['valence'] < 0.4) | (df_final['valence'] > 0.6)].copy()
    
    df_model['emotion'] = np.where(df_model['valence'] > 0.6, 1, 0)
    df_model['rotulo_emocao'] = np.where(df_model['valence'] > 0.6, 'Feliz', 'Triste')
    
    features = ['tempo', 'energy', 'loudness']
    
    df_train, df_test = train_test_split(df_model, test_size=0.3, random_state=42)
    
    X_train = df_train[features]
    y_train = df_train['emotion']
    X_test = df_test[features]
    y_test = df_test['emotion']

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "="*40)
    print("--- RESULTADOS DO MODELO ---")
    print("="*40)
    print(classification_report(y_test, y_pred))
    
    print("\n" + "="*40)
    print("--- PREVISÕES PARA SUAS MÚSICAS ---")
    print("="*40)
    resultados = df_test[['track_name', 'rotulo_emocao']].copy()
    resultados['Previsao_Modelo'] = ['Feliz' if p == 1 else 'Triste' for p in y_pred]
    resultados['Probabilidade_%'] = (y_proba * 100).round(2)
    
    print(resultados.head(15).to_string(index=False))

    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(x='energy', y='emotion', data=df_model, color='black', alpha=0.5, label='Dados Reais')
    
    sns.regplot(x='energy', y='emotion', data=df_model, logistic=True, ci=None, 
                scatter=False, color='red', line_kws={'label': 'Curva de Probabilidade (Modelo)'})
    
    plt.title('Regressão Logística: Probabilidade de ser "Feliz" baseada na Energia', fontsize=14)
    plt.ylabel('Probabilidade (0=Triste, 1=Feliz)', fontsize=12)
    plt.xlabel('Energy (Energia da Música)', fontsize=12)
    plt.yticks([0, 0.5, 1], ['0% (Triste)', '50% (Dúvida)', '100% (Feliz)'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

else:
    print("Poucas músicas encontradas. Aumente ainda mais o .head() ou verifique os nomes.")