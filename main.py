#!/usr/bin/env python
# coding: utf-8

# In[3]:


from nba_api.stats.endpoints import leaguegamefinder, playercareerstats
from nba_api.stats.static import players
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import statsmodels.api as sm
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR


# Test pour Nikola Jokić (player_id = '203999')

# In[4]:


NikoJokic = playercareerstats.PlayerCareerStats(player_id='203999')
jokic_df = NikoJokic.get_data_frames()[0]
print(jokic_df)


# Liste de tous les joueurs NBA sous forme de liste avec nom/prénom/full nom/id/is_active
# On Récupère tous les joueurs NBA (actifs + historiques)

# In[5]:


nba_players = players.get_players()


# On transforme le csv des salaires des joueurs et du salary cap en DataFrame pandas

# In[6]:


sal_df = pd.read_csv('data/Salaries.csv', delimiter=';') 
sal_df['Season'] = sal_df['Season'].apply(lambda x: f"{x-1}-{str(x)[-2:]}")
print(sal_df[sal_df["Season"] == "2024-25"].head(20))


cap_df = pd.read_csv('data/salary_cap.csv', delimiter=';')
print("\n", cap_df)


# On fait la jointure sur la colonne 'Saison'

# In[7]:


sal_cap_df = sal_df.merge(cap_df, on='Season', how='left')
sal_cap_df = sal_cap_df[sal_cap_df["Season"] != "1999-00"]
sal_cap_df["ratio_cap"] = sal_cap_df["Salary"] / sal_cap_df["SalaryCapUSD"]
sal_cap_df = sal_cap_df.rename(columns={"Season": "SEASON_ID"})
sal_cap_df["Season"] = sal_cap_df["SEASON_ID"].astype(str)
print(sal_cap_df[sal_cap_df["Season"] == "2024-25"].head(20))
#print(sal_cap_df["Season"].dtype)


# On a les 25 csv avec les stats des joueurs. On les mets en df et on les travaille 

# In[8]:


for i in range(25):
    if i < 10:
        var_name = f"stats_df_0{i}"
    else:
        var_name = f"stats_df_{i}"
    globals()[var_name] = pd.read_csv(f'data/nba_players{i}.csv')
    globals()[var_name].drop(columns=['Player.1'], inplace=True) #Il y a une deuxième colonne nom qui apparaît, je ne sais pas pourquoi. Je peux donc la drop
    globals()[var_name] = globals()[var_name][globals()[var_name]['SEASON_ID'] != '2025-26'] #On drop la saison 2025-26 dans chaque df parce qu'elle est incomplète
    list_drop = []
    mask = (~globals()[var_name].duplicated(subset=['Player', 'SEASON_ID'], keep=False)| (globals()[var_name]['TEAM_ABBREVIATION'] == 'TOT'))
    globals()[var_name] = globals()[var_name][mask].reset_index(drop=True)
    globals()[var_name] = globals()[var_name].sort_values(['Player','SEASON_ID']).reset_index(drop=True)
    for j in range(len(globals()[var_name]) - 1):
        if globals()[var_name]['GP'][j] < 30 :
            list_drop.append(j)
    globals()[var_name].drop(index=list_drop).reset_index(drop=True)   
    if 'Unnamed: 0' in globals()[var_name].columns:
        globals()[var_name].drop(columns=['Unnamed: 0'], inplace=True) #On drop la colonne Unnamed: 0 qui apparaît à la lecture du csv


print(stats_df_09.columns)
print(stats_df_09.head(20))


# On vérifie la présence de valeurs manquantes dans le df. On compte le nombre de zéros dans chaque colonne numérique. Des zéros en PPG, RPG, APG, GP, FG_PCT, FTA, FTM, PFPG montrent que le joueur n'a pas (ou à peine) joué. On ne peut pas l'inclure dans le modèle car ces zéros représentent un manque flagrant d'absence de données sur le joueur.
#  

# In[9]:


for i in range(25):
    if i < 10 : 
        var_name = f"stats_df_0{i}"
    else:
        var_name = f"stats_df_{i}"
    #print(globals()[var_name].isna().sum())
    #print(globals()[var_name][globals()[var_name].isna().any(axis=1)])

    #On compte le nombre de zéros dans chaque colonne numérique. Des zéros en PPG, RPG, APG, GP, FG_PCT, FTA, FTM, PFPG montrent que le joueur n'a pas ou à peine joué. On ne peut pas l'inclure dans le modèle car ces zéros représentent un manque flagrant d'absence de données sur le joueur
    colonnes_critiques = ["PPG", "RPG", "APG", "GP", "FG_PCT", "FTA", "FTM", "PFPG"]
    globals()[var_name] = globals()[var_name][~(globals()[var_name][colonnes_critiques] == 0).any(axis=1)] #On enlève les lignes qui contiennt un 0 parmi une des "stats critiques"
    #for col in colonnes_critiques:
        #if globals()[var_name][col].dtype in ['float64', 'int64']:
            #print(f"{col}: { (globals()[var_name][col] == 0).sum() } zéros")



# On fusionne les 25 df des stats des joueurs en un seul df

# In[10]:


liste_stats_df = [globals()[f"stats_df_0{i}"] for i in range(10)]+[globals()[f"stats_df_{i}"] for i in range(10, 25)]
stats_all_df = pd.concat(liste_stats_df, ignore_index=True)
print(f"On a {len(stats_all_df)} saisons de joueurs après nettoyage dans le dataset")  


# On fusionne les deux dataframes que l'on avait avant pour ne travailler qu'avec la fusion (full_df)

# In[11]:


# Merge X et Y

full_df = stats_all_df.merge(sal_cap_df[["Player","SEASON_ID","ratio_cap"]], on=["Player","SEASON_ID"], how='inner')
print(full_df.head(20))

#On trace la distributuon de certaines statistiques pour en avoir une meilleure idée

stats = ['PPG', 'APG', 'RPG', "ratio_cap"]
for stat in stats:
    sns.histplot(full_df[stat], kde=True, bins=30)
    plt.title(f"Distribution de {stat}")
    plt.show()

# Etude statistique du df : statistiques descriptives et intervalle de confiance 95%

# In[12]:


y = full_df['ratio_cap']

# Statistiques descriptives de base
print(y.describe())  # count, mean, std, min, quartiles, max [web:547][web:543]

# Intervalle de confiance à 95 % pour la moyenne de y
mean = y.mean()
std = y.std(ddof=1)
n = y.count()
se = std / np.sqrt(n)
z = 1.96  # pour 95%
ci_low = mean - z * se
ci_high = mean + z * se
print(f"IC 95 % pour la moyenne de ratio_cap : [{ci_low:.4f}, {ci_high:.4f}]")


# Corrélation

# In[13]:


for elt in full_df.columns : 
    if elt not in ['Player', 'SEASON_ID', 'TEAM_ABBREVIATION', 'ratio_cap']:
        print(full_df[['ratio_cap',elt]].corr())


# Illustration de la relation entre PPG ratio_cap

# In[18]:


for elt in full_df.columns :
    if elt not in ['Player', 'SEASON_ID', 'TEAM_ABBREVIATION', 'ratio_cap']:
        sns.regplot(x=elt, y='ratio_cap', data=full_df, ci=None)  
        plt.xlabel(f'{elt}')
        plt.ylabel('Ratio salaire / salary cap')
        plt.title(f'Relation entre {elt} et ratio de salaire')
        plt.show()


# In[ ]:


X = full_df[['PPG']]          
X = sm.add_constant(X)    # On ajoute la constante pour l’intercept
y = full_df['ratio_cap']

model = sm.OLS(y, X).fit()
print(model.summary())


# D'abord on initialise le spliter gss1. On prend 20% de nos données pour le test (et donc 80% pour entraîner et valider le modèle). Le random_state = 42 est une seed qui permet donc de reproduire l'expérience si on le souhaite (ou la changer).
# ggs1.split retourne les indices des lignes qui vont en train/validation ou en test. 
# test est donc le set de test (on a récupéré test en utilisant les indices contenus dans test_idx). De même avec trainval. 
# 
# gss2 sert à faire le split en training et validation. On reprend encore une fois 20% pour la validation. On change la seed à 43. On ne réutilise pas gss1 pour qu'il y ait indépendance totale des processus de sélection des sets de données.
# On récupère train et val de la même manière que l'on a pris test et trainval précédement. 

# In[ ]:


groups = full_df['Player']

gss1 = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42) # Test set (20%)
trainval_idx, test_idx = next(gss1.split(full_df, full_df['ratio_cap'], groups))
trainval = full_df.iloc[trainval_idx].reset_index(drop=True)
test = full_df.iloc[test_idx].reset_index(drop=True)

gss2 = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=43) # Validation set (16% tot)
train_idx, val_idx = next(gss2.split(trainval, trainval['ratio_cap'], trainval['Player']))
train = trainval.iloc[train_idx].reset_index(drop=True)
val = trainval.iloc[val_idx].reset_index(drop=True)


# Ici on utilise des pipelines. les pipelines servent à chaîner le pré-traitement et l'utilisation du modèle de manière concise et "automatique" (On peut les réutiliser ce qui est pratique pour accélérer et uniformiser le travail.
# num_proc sert à traiter les colonnes numériques. SimpleImputer sert à insérer la médiane dans des colonnes où des valeurs manquent. Scaler permet "d'uniformiser" les valeurs en ayant une moyenne à 0 (ou 1 je sais plus) et 1 d'écart-type. Cela aide pour les calculs.
# 
# Le dernier pipeline contient ridge comme modèle. Ridge est le modèle que nous utilisons. Il s'agit d'une régression multi-linéaire avec une régularisation de termes au carré. C'est une régression multi-linéaire avec une fonction de coût de la forme (somme des lambda alpha^2) qui sera minimisée. Cela nous est très utile dans notre cas car il y a énormément de statistiques en NBA et il fait donc sens de les inclure comme paramètres de notre modèle. Cependant, il risque fatalement de s'avérer que plusieurs de ces statistiques n'ont pas une grande importance dans le modèle final. Le ridge permet de minimiser les paramètres qui ont un minimum d'importance. Il existe aussi un modèle qui permet d'annuler l'importance de certains paramètres (Lasso). Nous préférons utiliser le ridge qui minimise sans annuler. 

# In[ ]:


# Définir les features (exclut Player, SEASON_ID, ratio_cap)
features = [c for c in train.columns if c not in ['Player', 'SEASON_ID', 'ratio_cap']]
num_cols = train[features].select_dtypes(include=['number']).columns.tolist()
cat_cols = [c for c in features if c not in num_cols]

num_proc = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_proc = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preproc = ColumnTransformer([
    ('num', num_proc, num_cols),
    ('cat', cat_proc, cat_cols)
])

ridge = Ridge(alpha=1.0) 
pipe = Pipeline([
    ('preproc', preproc),
    ('model', ridge)
])


# On initialise un GroupKFold avec 5 sous-ensembles. ON utilise la méthode GroupKFold plutôt qu'un KFold calssique puisque former des groupes est pertinents dans notre cas. Ne pas faire des groupes contenant à chaque fois toutes les saisons d'un même joueur causerait une "fuite de données" (data leakage) entre train set et test set. Prenons l'exemple de Lebron James. Toutes ses saisons (toutes les lignes contenant "Lebron James" dans la colonne Player) seront soit en test set soit dans le train set (et dans le même sous-ensemble, ce qui fait qu'on ne peut pas avoir Lebron James et dans le train set et dans le validation set). Le seul inconvénient est que l'on a un processus moins randomisé que le Kfold qui en forme aucun group. On aura une variance un peu plus élevé mais on évite tout data leakage et on donne du sens à chaque set. 
# 
# Ensuite, groups_train indique dans quel ensemble sont chaque joueur. 
# 
# alphas est un dictionnaire qui représente la grille d'hyperparamètre pour GridSearchCV. les valeurs de alpha sont les paramètres de régularisation du ridge. 
# 
# GridSearchCV prend en arguments alphas, la méthode d'évaluation (moyenne de l'erreur en valeur absolue) et les features/labels du group d'entraînement oragnisé avec gfk. L'objectif est de tester le modèle avec tous les hyperparamètres alphas différents. Ils sont ensuite évalués selon la méthode d'évaluation. Le alpha qui permet de minimiser le scoring est maintenu. 
# A titre d'exemple, si on prenaint la méthode des KNN, GridSearchCV nous permettrait de savoir le nombre optimal de voisins à avoir pour faire la meilleure prédiction possible. 
# 
# Enfin, on fit notre modèle avec les features et les labels. 

# In[ ]:


gkf = GroupKFold(n_splits=5)
groups_train = train['Player']

# GridSearch pour ajuster alpha (régularisation) sur le train set
alphas = {'model__alpha': [0.01, 0.1, 1, 10, 100]}
gs = GridSearchCV(pipe, param_grid=alphas, scoring='neg_mean_absolute_error', cv=gkf.split(train[features], train['ratio_cap'], groups_train))
gs.fit(train[features], train['ratio_cap'])

print("Meilleur alpha:", gs.best_params_)
model_fitted = gs.best_estimator_.named_steps['model']  # extrait l'objet Ridge
pipe = gs.best_estimator_
preproc_fitted = pipe.named_steps['preproc']
model_fitted = pipe.named_steps['model']  # si tu n'utilises pas MultiOutputRegressor


feature_names = preproc_fitted.get_feature_names_out()
for name, coef in zip(feature_names, model_fitted.coef_):
    print(f"{name}: {coef}")
print("Intercept :", model_fitted.intercept_)

print("Score de CV (MAE):", -gs.best_score_)


# In[ ]:


val_preds = gs.predict(val[features])
mae_val = mean_absolute_error(val['ratio_cap'], val_preds)
print(f"MAE validation set : {mae_val:.4f}")
r2 = r2_score(test['ratio_cap'], test_preds)
print("R2 Test Lasso :", r2)
n = test.shape[0]       # nombre de lignes (échantillons)
p = test.shape[1]       # nombre de colonnes (features)
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("R² ajusté :", r2_adj)


# In[ ]:


trainval = pd.concat([train, val]).reset_index(drop=True)
pipe_best = Pipeline([
    ('preproc', preproc),
    ('model', Ridge(alpha=gs.best_params_['model__alpha']))
])
pipe_best.fit(trainval[features], trainval['ratio_cap'])
test_preds = pipe_best.predict(test[features])
mae_test = mean_absolute_error(test['ratio_cap'], test_preds)
print(f"MAE test set : {mae_test:.4f}")
print("MAE Validation:", mean_absolute_error(val['ratio_cap'], val_preds))
print("MAE Test:", mean_absolute_error(test['ratio_cap'], test_preds))
moy = test['ratio_cap'].mean()
err_moy = mean_absolute_error(test['ratio_cap'], test_preds)
print(moy, err_moy)


# On obtient une erreur moyenne absolue de 0,0391 sur la prédiction. On est proche de la même erreur entre test set et validation set. On a donc évité l'overfitting. 
# 
# 
# Désormais, on se penche sur un deuxième modèle, plus strict : le Lasso. Il s'agit aussi d'une régression mutli-linéaire mais celle-ci peut annuler certains paramètres grâce à la fonction de coût que l'on ajoute à la régression multi-linéaire (l'objectif étant toujours de minimiser cette fonction de coût). 

# In[ ]:


lasso = Lasso(max_iter=10000) 
pipe_l = Pipeline([
    ('preproc', preproc),
    ('model', lasso)
])
alphas = {'model__alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10, 30, 100]}
gkf = GroupKFold(n_splits=5)
groups_train = train['Player']

gs = GridSearchCV(pipe_l, param_grid=alphas,
                  scoring='neg_mean_absolute_error',
                  cv=gkf.split(train[features], train['ratio_cap'], groups_train))
gs.fit(train[features], train['ratio_cap'])

print("Meilleur alpha:", gs.best_params_)
print("Score de CV (MAE):", -gs.best_score_)


# In[ ]:


lasso_fitted = gs.best_estimator_.named_steps['model']
preproc_fitted = gs.best_estimator_.named_steps['preproc']
feature_names = preproc_fitted.get_feature_names_out()

for name, coef in zip(feature_names, lasso_fitted.coef_):
    print(f"{name}: {coef}")


# On voit qu'il a été intéressant de se pencher sur le lasso après le ridge tant une partie importante des paramètres a été annulée par ce modèle. Avec le lasso, on se détac he de l'équipe d'origine du joueur ainsi que les statistiques relatives au nombre de paniers marqués/tentés. 
# 
# On peut trouver une cohérence à ce phénomène. Pour les équipes, elles ont certes chacune des particularités qui font que l'on ppurrait juger important de les compter parmi les paramètres, mais entre 2000 et 2025, chaque équipe (plus ou moins) est passée par différentes phases (en terme de compétitivité au sein de la ligue) avec le besoin de recourir à la totalité du cap salarial (ou plus) pour payer les joueurs ainsi qu'à des phases de basse compétitivité où les équipes n'usent pas forcément de la totalité du cap salarial (dans le but de faire des économies).
# 
# Pour les statistiques de tirs tentés et marqués, on peut expliquer leur absence par le fait qu'elles se rapportent de manière plutôt proche à la statistique de points pas match (PPG) qui est la plus coefficientée dans le lasso.  

# In[ ]:


val_preds = gs.predict(val[features])
test_preds = gs.predict(test[features])
print("MAE Validation:", mean_absolute_error(val['ratio_cap'], val_preds))
print("MAE Test:", mean_absolute_error(test['ratio_cap'], test_preds))
print("MAE Validation:", mean_absolute_error(val['ratio_cap'], val_preds))
print("MAE Test:", mean_absolute_error(test['ratio_cap'], test_preds))
moy = test['ratio_cap'].mean()
err_moy = mean_absolute_error(test['ratio_cap'], test_preds)
print(moy, err_moy)


# On a 0,0365 d'erreur moyenne absolue pour le validation set et 0,0397 pour le test set. On a donc pas d'overfitting
# Cependant, on voit que le lasso a une plus grande erreur que le ridge (0,0391 contre 0,0397). On serait donc plutôt tenté de prendre le ridge.
# 
# 

# Désormais, on veut tester un modèle non-linéaire pour voir s'il performe mieux que nos 2 modèle multi-linéaires. On va utiliser le DecisionTreeRegressor.

# In[ ]:


tree = DecisionTreeRegressor(random_state=42)
pipe_t = Pipeline([
    ('preproc', preproc),
    ('model', tree)])
param_grid = {'model__max_depth': [3, 5, 7, 10, 15, None]}
gkf = GroupKFold(n_splits=5)
groups_train = train['Player']

gs = GridSearchCV(pipe_t, param_grid=param_grid,
                  cv=gkf.split(train[features], train['ratio_cap'], groups_train),
                  scoring='neg_mean_absolute_error')
gs.fit(train[features], train['ratio_cap'])

print("Meilleur max_depth :", gs.best_params_["model__max_depth"])
print("Score de CV (MAE):", -gs.best_score_)


# In[ ]:


best_tree = gs.best_estimator_.named_steps['model']
preproc_fitted = gs.best_estimator_.named_steps['preproc']
feature_names = preproc_fitted.get_feature_names_out()

importances = best_tree.feature_importances_
for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"{name}: {imp:.4f}")


# In[ ]:


val_preds_tree = gs.predict(val[features])
test_preds_tree = gs.predict(test[features])
print("MAE Validation:", mean_absolute_error(val['ratio_cap'], val_preds_tree))
print("MAE Test:", mean_absolute_error(test['ratio_cap'], test_preds_tree))
r2 = r2_score(test['ratio_cap'], test_preds_tree)
print("R2 Test Tree :", r2)
n = test.shape[0]       # nombre de lignes (échantillons)
p = test.shape[1]       # nombre de colonnes (features)
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("R² ajusté :", r2_adj)


# On a un meilleur résultat pour le DecisionTreeRegressor. Une erreur moyenne absolue de 0,036 comparée aux 0,0391 et 0,0397 des deux modèles multi-linéaires. 
# On peut donc être tentés de choisir ce modèle instinctivement mais il ne faut pas oublier que ce modèle non-linéaire n'a pas "l'explicativité" des deux modèles multi-linéaires. On ne peut pas vraiment dire : "le coefficient PPG est pondéré à une valeur de XXX donc quand le PPG monte de YYY, le salaire monte de ZZZ". 
# A noter qu'on a un r2 de 0,58, ce qui peut paraitre mauvais. Cependant, au vu des résultats obtenus par des professionnels (r2 max de 0,63 avec du svm https://towardsdatascience.com/predicting-nba-salaries-with-machine-learning-ed68b6f75566/), on peut estimer que c'est finalement acceptable

# On essaie d'utiliser le modèle svr pour voir si on a un meilleur r2

# In[ ]:


svr = SVR(kernel='rbf')  # kernel RBF par défaut, non linéaire

pipe_svr = Pipeline([
    ('preproc', preproc),
    ('model', svr)
])


# In[ ]:


param_grid = {
    'model__C': [1, 10],
    'model__epsilon': [0.05, 0.1],
    'model__gamma': ['scale', 'auto']  
}

gkf = GroupKFold(n_splits=5)
groups_train = train['Player']

gs_svr = GridSearchCV(
    pipe_svr,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=gkf.split(train[features], train['ratio_cap'], groups_train),
    n_jobs=-1
)

gs_svr.fit(train[features], train['ratio_cap'])

print("Meilleurs hyperparamètres SVR :", gs_svr.best_params_)
print("MAE CV (moyen) :", -gs_svr.best_score_)


# In[ ]:


y_val_pred = gs_svr.predict(val[features])
mae_val = mean_absolute_error(val['ratio_cap'], val_preds)
print("MAE validation SVR :", mae_val)

# Test
y_test_pred_svr = gs_svr.predict(test[features])
mae_test = mean_absolute_error(test['ratio_cap'], y_test_pred_svr)
from sklearn.metrics import r2_score
r2_test = r2_score(test['ratio_cap'], y_test_pred_svr)
print("MAE test SVR :", mae_test)
print("R2 test SVR  :", r2_test)
n = test.shape[0]       # nombre de lignes (échantillons)
p = test.shape[1]       # nombre de colonnes (features)
r2_adj = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)
print("R² ajusté :", r2_adj)

