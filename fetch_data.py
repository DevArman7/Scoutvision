import soccerdata as sd
import pandas as pd

# Fetch seasonal player stats from FBref for the Premier League & La Liga
fbref = sd.FBref(leagues=['ENG-Premier League', 'ESP-La Liga'], seasons='2324')

# Pull standard, shooting, and passing stats
standard_stats = fbref.read_player_season_stats(stat_type='standard')

# Save directly to CSV
standard_stats.to_csv("realplayers.csv")
print("Data extracted successfully!")