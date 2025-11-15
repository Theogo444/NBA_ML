from nba_api.stats.endpoints import leaguegamefinder, playercareerstats
from nba_api.stats.static import players
import pandas as pd
import time


# Test pour Nikola Jokić (player_id = '203999')
# NikoJokic = playercareerstats.PlayerCareerStats(player_id='203999')
# jokic_df = NikoJokic.get_data_frames()[0]
# print(jokic_df)


#Liste de tous les joueurs NBA sous forme de liste avec nom/prénom/full nom/id/is_active
#On Récupère tous les joueurs NBA (actifs + historiques)

custom_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/"
}


sample_size = 200
nba_players = players.get_players()



#On initialise une liste pour stocker les DataFrames
all_stats = []

count_error = 0
joueurs_error = []

# On boucle sur chaque joueur actif (pour l'instant on écarte les retraités)
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

df_all_players.to_csv('nba_players.csv', index=False)

#print(f"\nDataFrame final : {df_all_players.shape}")
#print(df_all_players.head())