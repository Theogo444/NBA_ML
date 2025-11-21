#!/usr/bin/env python
# coding: utf-8

# In[2]:


from nba_api.stats.endpoints import leaguegamefinder, playercareerstats
from nba_api.stats.static import players
import pandas as pd
import time


# Test pour Nikola Jokić (player_id = '203999')

# In[15]:


NikoJokic = playercareerstats.PlayerCareerStats(player_id='203999')
jokic_df = NikoJokic.get_data_frames()[0]
print(jokic_df)


# Liste de tous les joueurs NBA sous forme de liste avec nom/prénom/full nom/id/is_active
# On Récupère tous les joueurs NBA (actifs + historiques)
# On utilise un sample_size = 200 car au delà de 200 et quelques requêtes, l'api bloque nos requêtes (erreur 403 : Forbiden)
# Ainsi, en raison de ce blocage au bout de 200 et quelques requêtes, on va boucler le programme pour obtenir les données des joueurs. Cependant, nous avons pu expérimenter que si nous faisons la boucle de manière classique, le programme nous bloque après la première boucle (avec n = sample_size itérations). Ainsi, nous avons rusé et nous avons donc fait une boucle mais de manière manuelle.
# Le processus est : i = 0, on exécute le programme, on récupère une liste de df de joueurs, on concatène. On a donc un df nommé df_all_players et on le transforme en csv (nba_players_i.csv) pour pouvoir stocker ces données sur l'ordinateur. Ainsi, on a les données de tous les joueurs que l'on souhaite séparées en 25 csv.  
# 
# Dans le même temps, on recense les erreurs pour pouvoir les analyser et les compter. Les erreurs viennent en grande majorité de joueurs qui sont renseignés comme étant en nba mais qui n'ont dans les fait pas joué de match nba (par exemple le rookie français Noa Essengué qui attend encore de joueur avec les chicago bulls même s'il est déjà dans l'effectif). 

# In[3]:


sample_size = 200
nba_players = players.get_players()



#On initialise une liste pour stocker les DataFrames
all_stats = []

count_error = 0
joueurs_error = []

# On boucle sur chaque joueur actif (on écarte les retraités)

Count_ex = int(len(nba_players)/sample_size)
print(Count_ex)
i = 0

sample_nba_players = nba_players[sample_size*i:sample_size*(i+1)]

for player in sample_nba_players:
    player_id = player['id']
    player_name = player['full_name']

    try:
        # Appel API pour les stats des joueurs
        career = playercareerstats.PlayerCareerStats(player_id=player_id)
        df = career.get_data_frames()[0]
        if int((df["SEASON_ID"][0])[5:]) < 26: #Filtrer les joueurs ayant commencé leur carrière NBA à partir de 2000 (< 26 ca veut dire qu'on va de 0 à 25 pour la première saison)
            # Premier DataFrame = stats par saison
            # Ajouter le nom du joueur dans le DataFrame
            df['Player'] = player_name
            # On met les stats PAR MATCH (Per Game = PG) parce que c'est ça qui est parlant (avec le nombre de matchs joués)
            # On arrondit à 2 décimales pour les stats classiques, 1 pour les fautes
            df['PPG'] = (df['PTS'] / df['GP']).round(2)
            df['RPG'] = (df['REB'] / df['GP']).round(2)
            df['APG'] = (df['AST'] / df['GP']).round(2)
            df['SPG'] = (df['STL'] / df['GP']).round(2)
            df['BPG'] = (df['BLK'] / df['GP']).round(2)
            df['TOVPG'] = (df['TOV'] / df['GP']).round(2)
            df['PFPG'] = (df['PF'] / df['GP']).round(1)
            df['FG3MPG'] = (df['FG3M'] / df['GP']).round(2)
            df['FG3APG'] = (df['FG3A'] / df['GP']).round(2)
            df['FGMPG'] = (df['FGM'] / df['GP']).round(2)
            df['FGAPG'] = (df['FGA'] / df['GP']).round(2)
            df['OREBPG'] = (df['OREB'] / df['GP']).round(2)


            # On enlève les colonnes inutiles
            df = df.drop('PTS', axis=1)
            df = df.drop('REB', axis=1)
            df = df.drop('OREB', axis=1)
            df = df.drop('DREB', axis=1)
            df = df.drop('AST', axis=1)
            df = df.drop('STL', axis=1)
            df = df.drop('BLK', axis=1)
            df = df.drop('TOV', axis=1)
            df = df.drop('PF', axis=1)
            df = df.drop('FG3M', axis=1)
            df = df.drop('FG3A', axis=1)
            df = df.drop('FGM', axis=1)
            df = df.drop('FGA', axis=1)
            df = df.drop('LEAGUE_ID', axis=1)
            df = df.drop('TEAM_ID', axis=1)

            # On réorganise les colonnes
            df = df[['Player', 'SEASON_ID', 'PPG', 'RPG', 'APG', 'BPG', 'SPG'] + [col for col in df.columns if col not in ['PLAYER_ID', 'SEASON_ID', 'PPG', 'RPG', 'APG', 'BPG', 'SPG']]]

            print(df)

            #Ajouter à la liste
            all_stats.append(df)

            print(f"✓ {player_name} récupéré")

    except Exception as e:
        count_error += 1
        joueurs_error.append(player_id)
        print(f"✗ Erreur pour {player_name}: {e}")
        continue

        #Pause pour éviter de surcharger l'API
    time.sleep(0.5)
    print(f"% de joueurs sélectionnés : {len(all_stats)/len(nba_players)*100:.2f} % \n Taille de l'échantillon : {len(nba_players)} \n Nombre d'erreurs : {count_error}", end='\r')


#print(joueurs_error)

#On Concatène tous les DataFrames
df_all_players = pd.concat(all_stats, ignore_index=True)

df_all_players.to_csv('nba_players_00.csv', index=False)

print(f"\nDataFrame final : {df_all_players.shape}")
print(df_all_players.head())

