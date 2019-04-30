import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from web_scraper_tools import request_zipped_url
from retrosheet_tools import gamelogs_to_df

# %% Load retrosheet data
df_retro = pd.concat([gamelogs_to_df(request_zipped_url(f'https://www.retrosheet.org/gamelogs/gl{yr}.zip')) for yr in range(2014, 2019)]
                     , axis='rows')

# %% Select data
vis_stats = ['VisAB', 'VisH', 'VisD', 'VisT', 'VisHR', 'VisSF', 'VisK']
hm_stats = ['HmAB', 'HmH', 'HmD', 'HmT', 'HmHR', 'HmSF', 'HmK']

df_retro_trim = (df_retro
                 .loc[:, ['ParkID', 'DayNight', 'VisTm', 'HmTm'] + vis_stats + hm_stats]
                 .eval('Year = index.dt.year')
                 .eval('Month = index.dt.month')
                 .eval('DOY = index.dt.dayofyear / 365')
                 )

# Vis team stats
df_vis = (df_retro_trim[vis_stats]
          .eval('VisBIP = VisAB - VisK + VisSF')  # Balls in play
          .eval('VisS = VisH - VisD - VisT - VisHR')  # Singles
          )

df_vis = (df_vis[['VisS', 'VisD', 'VisT', 'VisHR']]
          .divide(df_vis['VisBIP'], axis='rows')  # % of S,D,T,HR per BIP
          .rename({'VisS': 'Singles', 'VisD': 'Doubles', 'VisT': 'Triples', 'VisHR': 'HR'}, axis='columns')
          )
df_vis['Year'] = df_retro_trim['Year']
df_vis['Month'] = df_retro_trim['Month']
df_vis['DOY'] = df_retro_trim['DOY']
df_vis['ParkID'] = df_retro_trim['ParkID']
df_vis['Team'] = df_retro_trim['VisTm']
df_vis['Loc'] = 'Away'


# Hm team stats
df_hm = (df_retro_trim[hm_stats]
         .eval('HmBIP = HmAB - HmK + HmSF')
         .eval('HmS = HmH - HmD - HmT - HmHR')
         )

df_hm = (df_hm[['HmS', 'HmD', 'HmT', 'HmHR']]
         .divide(df_hm['HmBIP'], axis='rows')
         .rename({'HmS': 'Singles', 'HmD': 'Doubles', 'HmT': 'Triples', 'HmHR': 'HR'}, axis='columns')
         )
df_hm['Year'] = df_retro_trim['Year']
df_hm['Month'] = df_retro_trim['Month']
df_hm['DOY'] = df_retro_trim['DOY']
df_hm['ParkID'] = df_retro_trim['ParkID']
df_hm['Team'] = df_retro_trim['HmTm']
df_hm['Loc'] = 'Home'

print(pd.concat([df_vis.describe(include=[np.float]).add_suffix('_vis').T
                    , df_hm.describe(include=[np.float]).add_suffix('_hm').T], axis='rows'))

# Combine Vis and Hm into one df
df_park = pd.concat([df_vis, df_hm], axis='rows', sort=False)

# OHE location
df_park = pd.concat([df_park, pd.get_dummies(df_park['Loc'])], axis='columns').drop(columns=['Loc'])


# %% Normalize stats by team by year
df_baseline = (df_park
               .groupby(['Year', 'Team'])[['Singles', 'Doubles', 'Triples', 'HR']]
               .agg(['mean', 'std'])
               )

# Combine column multiindex into a single set of column names
df_baseline.columns = df_baseline.columns.map('_'.join)

# Rest index so year and team are now columns
df_baseline.reset_index(inplace=True)

# Join with park data
df_park_normalized = (df_park
                      .merge(df_baseline, on=['Year', 'Team'])
                      .eval('Singles_norm = (Singles - Singles_mean)/Singles_std')
                      .eval('Doubles_norm = (Doubles - Doubles_mean)/Doubles_std')
                      .eval('Triples_norm = (Triples - Triples_mean)/Triples_std')
                      .eval('HR_norm = (HR - HR_mean)/HR_std')
                      )

# Park counts
park_counts = (df_park_normalized['ParkID']
               .value_counts()
               .to_frame()
               .reset_index()
               .rename({'index': 'ParkID', 'ParkID': 'Count'}, axis='columns')
               )

# Park dict
parkID_dict = dict([(park, i) for i, park in enumerate(park_counts['ParkID'])])
parkINT_dict = dict([(i, park) for park, i in parkID_dict.items()])

# Filter out infrequent parks
infreq_parks = park_counts.query('Count < 30').loc[:, 'ParkID'].to_list()
df_park_normalized = df_park_normalized.query('ParkID not in @infreq_parks')

# Replace ParkID with ParkID_int
df_park_normalized['ParkID'] = [parkID_dict.get(parkID) for parkID in df_park_normalized['ParkID']]

# Determine home parks
home_parks = (df_park_normalized
              .query('Home == 1')
              .groupby(['Year', 'Team'])
              .last()
              .loc[:, 'ParkID']
              .reset_index()
              .rename({'ParkID': 'HomeParkID'}, axis='columns')
              )

# Join with park data
df_park_normalized = df_park_normalized.merge(home_parks, on=['Year', 'Team'])

# Visualize distributions
fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True)
wts = np.ones(df_park_normalized.shape[0]) / df_park_normalized.shape[0]
df_park_normalized['Singles_norm'].plot(ax=axs[0], kind='hist', weights=wts)
df_park_normalized['Doubles_norm'].plot(ax=axs[1], kind='hist', weights=wts)
df_park_normalized['Triples_norm'].plot(ax=axs[2], kind='hist', weights=wts)
df_park_normalized['HR_norm'].plot(ax=axs[3], kind='hist', weights=wts)
plt.show()


fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
df_park_normalized.query('ParkID == "DEN02"').loc[:, 'Doubles_norm'].plot(ax=axs[0], kind='hist')
df_park_normalized.query('ParkID == "PIT08"').loc[:, 'Doubles_norm'].plot(ax=axs[1], kind='hist')
plt.show()


# %% Train, test splits
df_training = df_park_normalized.query('Year < 2018')
df_testing = df_park_normalized.query('Year >= 2018')

df_training.describe(include=[np.number]).T

# %% Compile Keras model
optim = keras.optimizers.Adam(lr=0.00005)

# Park Factor Embedding
n_parks = df_training['ParkID'].nunique()
n_embedding_features = 2

# --Create an input layer for the team ID
park_in = keras.layers.Input(shape=(1,))

# --Park factor lookup
park_factor_embedding = keras.layers.Embedding(input_dim=n_parks
                                               , output_dim=n_embedding_features
                                               , input_length=1
                                               , name='Park_Factor'
                                               , embeddings_initializer='uniform'
                                               , embeddings_regularizer=None
                                               , activity_regularizer=None
                                               , embeddings_constraint=None
                                               , mask_zero=False
                                               )
# --Lookup the park id in the park factor embedding layer
factor_lookup = park_factor_embedding(park_in)

# --Flatten embedding output
factor_lookup_flat = keras.layers.Flatten()(factor_lookup)

# --Combine into a single, re-usable model
park_factor_model = keras.models.Model(name='Park-Factor-Model', inputs=park_in, outputs=factor_lookup_flat)


# Merge input layers
game_park = keras.layers.Input(name='Game_Park', shape=(1,))
home_park = keras.layers.Input(name='Home_Park', shape=(1,))
context_info = keras.layers.Input(name='Context', shape=(2,))

# --Park factors
game_park_factor = park_factor_model(game_park)
home_park_factor = park_factor_model(home_park)

# --Concatenate
nn = keras.layers.Concatenate()([game_park_factor
                                                  , home_park_factor
                                                  , context_info])


# Dense hidden layers
# nn = keras.layers.Dense(name='Hidden_Layer_1', units=30, activation='relu')(full_input_layer)
# nn = keras.layers.Dropout(rate=0.25)(nn)
# nn = keras.layers.Dense(name='Hidden_Layer_2', units=3, activation='relu')(nn)
# nn = keras.layers.Dropout(rate=0.25)(nn)
# nn = keras.layers.Dense(name='Hidden_Layer_3', units=5, activation='relu')(nn)
# nn = keras.layers.Dropout(rate=0.25)(nn)



# Output layers
out_1 = keras.layers.Dense(name='Singles_Output', units=1, activation='linear')(nn)
out_2 = keras.layers.Dense(name='Doubles_Output', units=1, activation='linear')(nn)
out_3 = keras.layers.Dense(name='Triples_Output', units=1, activation='linear')(nn)
out_4 = keras.layers.Dense(name='HR_Output', units=1, activation='linear')(nn)


# Create final model
model = keras.models.Model(name='Park_Factor_Learning_Model'
                           , inputs=[game_park
                                    , home_park
                                    , context_info]
                           , outputs=[out_1, out_2, out_3, out_4]
                           )

# Compile with different loss functions and weights for losses
loss_funcs = {'Singles_Output': 'mean_absolute_error'
    , 'Doubles_Output': 'mean_absolute_error'
    , 'Triples_Output': 'mean_absolute_error'
    , 'HR_Output': 'mean_absolute_error'}
loss_weights = {'Singles_Output': 0, 'Doubles_Output': 0.5, 'Triples_Output': 0.0, 'HR_Output': 0.5}
model.compile(optimizer=optim, loss=loss_funcs, loss_weights=loss_weights)

# Model summary
print(model.summary())



# %% Train Keras model
history = model.fit(x=[df_training['ParkID']
                    , df_training['HomeParkID']
                        , df_training[['DOY', 'Home']]]
                    , y=[df_training['Singles_norm'], df_training['Doubles_norm'], df_training['Triples_norm'], df_training['HR_norm']]
                    , validation_split=0.10
                    , epochs=500
                    , batch_size=500
                    , verbose=2
                    )

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


park_factors = pd.DataFrame(model.get_weights()[0]).rename({0: 'PF1', 1: 'PF2'}, axis='columns')
park_factors['ParkID'] = [parkINT_dict.get(i) for i in park_factors.index]

fig, ax = plt.subplots()
park_factors.plot(ax=ax, kind='scatter', x='PF1', y='PF2')
for k, v in park_factors.iterrows():
    ax.annotate(v[2], (v[0], v[1]))
plt.show()

