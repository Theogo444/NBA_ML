from nba_api.stats.endpoints import leaguegamefinder, playercareerstats
from nba_api.stats.static import players
import pandas as pd
import time


nba_players = players.get_players()[:10]

NikoJokic = playercareerstats.PlayerCareerStats(player_id='203999')
df = NikoJokic.get_data_frames()[0]

id_error = [1629152, 1641851, 1824, 202079, 203951, 1642379, 1643113, 1630555, 1858, 1627760, 1628238, 1642926, 1641777]

career = playercareerstats.PlayerCareerStats(player_id=id_error[0])
df_error = career.get_data_frames()[1]
print(df_error)

#On met les stats PAR MATCH parce que c'est ça qui est parlant (avec le nombre de matchs joués)
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




# Réorganiser les colonnes
df = df[['PLAYER_ID', 'SEASON_ID', 'PPG', 'RPG', 'APG', 'BPG', 'SPG'] + [col for col in df.columns if col not in ['PLAYER_ID', 'SEASON_ID', 'PPG', 'RPG', 'APG', 'BPG', 'SPG']]]

#if int((df["SEASON_ID"][0])[5:]) <= 26: 
    #print("c'est ok") # Affiche l'année de la première saison jouée