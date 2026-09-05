import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "players.csv")

class ScoutingEngine:
    def __init__(self, csv_path: str = CSV_PATH):
        # 1. Load raw dataset
        raw_df = pd.read_csv(csv_path)
        
        # 2. Standardize column names to lowercase
        raw_df.columns = [c.strip() for c in raw_df.columns]
        
        # Mapping common Kaggle/FBref columns to clean internal names
        column_map = {
            'Player': 'name',
            'Squad': 'team',
            'Comp': 'league',
            'Pos': 'position',
            'Age': 'age',
            'Min': 'minutes',
            'Gls': 'goals',
            'Ast': 'assists',
            'xG': 'xg',
            'xAG': 'xa',
            'PrgP': 'progressive_passes',
            'PrgC': 'progressive_carries',
            'Tkl': 'tackles',
            'Int': 'interceptions'
        }
        
        # Rename available columns
        self.df = raw_df.rename(columns={k: v for k, v in column_map.items() if k in raw_df.columns}).copy()
        
        # Ensure an ID column exists
        if 'id' not in self.df.columns:
            self.df['id'] = range(1, len(self.df) + 1)

        # Simplify position (e.g., 'MF,FW' -> 'MF')
        if 'position' in self.df.columns:
            self.df['position'] = self.df['position'].astype(str).str.split(',').str[0].str.strip()

        # Clean numerical columns
        if 'minutes' in self.df.columns:
            self.df['minutes'] = pd.to_numeric(self.df['minutes'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)
            # Filter players with at least 180 minutes played for reliable stats
            self.df = self.df[self.df['minutes'] >= 180].reset_index(drop=True)

        # Identify numeric feature columns for similarity & radar
        self.feature_columns = []
        possible_metrics = ['goals', 'assists', 'xg', 'xa', 'progressive_passes', 'progressive_carries', 'tackles', 'interceptions']
        
        for metric in possible_metrics:
            if metric in self.df.columns:
                # Convert to numeric per-90 metrics if needed
                self.df[metric] = pd.to_numeric(self.df[metric], errors='coerce').fillna(0.0)
                # Calculate per 90 value
                p90_name = f"{metric}_p90"
                self.df[p90_name] = ((self.df[metric] / self.df['minutes']) * 90).round(2)
                self.feature_columns.append(p90_name)

        self.scaler = StandardScaler()
        self._preprocess()

    def _preprocess(self):
        # 1. Compute Percentiles per position group (0 to 100)
        for metric in self.feature_columns:
            pct_col = f"{metric}_pct"
            self.df[pct_col] = (
                self.df.groupby("position")[metric]
                .rank(pct=True) * 100
            ).round(1)

        # 2. Scale feature vectors (Z-score normalization)
        scaled_features = self.scaler.fit_transform(self.df[self.feature_columns])
        
        # 3. Precompute Cosine Similarity Matrix
        self.similarity_matrix = cosine_similarity(scaled_features)

    def search_players(
        self, 
        position: Optional[str] = None, 
        min_minutes: int = 0,
        query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        filtered = self.df.copy()
        
        if position:
            filtered = filtered[filtered["position"].str.upper() == position.upper()]
        if min_minutes > 0:
            filtered = filtered[filtered["minutes"] >= min_minutes]
        if query:
            filtered = filtered[filtered["name"].str.contains(query, case=False, na=False)]
            
        columns_to_return = ["id", "name", "team", "position", "minutes"]
        existing_cols = [c for c in columns_to_return if c in filtered.columns]
        return filtered[existing_cols].head(50).to_dict(orient="records")

    def get_player_profile(self, player_id: int) -> Optional[Dict[str, Any]]:
        player_row = self.df[self.df["id"] == player_id]
        if player_row.empty:
            return None
        
        row = player_row.iloc[0]
        
        # Build radar-ready metrics (Raw Value vs Percentile)
        stats = {}
        for feat in self.feature_columns:
            stats[feat] = {
                "raw_value": float(row[feat]),
                "percentile": float(row.get(f"{feat}_pct", 50.0))
            }
            
        return {
            "id": int(row["id"]),
            "name": str(row["name"]),
            "team": str(row.get("team", "Unknown")),
            "position": str(row.get("position", "N/A")),
            "minutes": int(row.get("minutes", 0)),
            "stats": stats
        }

    def get_similar_players(self, player_id: int, top_k: int = 5) -> Optional[List[Dict[str, Any]]]:
        if player_id not in self.df["id"].values:
            return None

        idx = self.df.index[self.df["id"] == player_id].tolist()[0]
        target_pos = self.df.loc[idx, "position"]

        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        similar_players = []
        for match_idx, score in sorted_scores:
            if match_idx == idx:
                continue
            
            # Position-matched comparison
            if self.df.loc[match_idx, "position"] == target_pos:
                similar_players.append({
                    "id": int(self.df.loc[match_idx, "id"]),
                    "name": str(self.df.loc[match_idx, "name"]),
                    "team": str(self.df.loc[match_idx].get("team", "Unknown")),
                    "position": str(self.df.loc[match_idx, "position"]),
                    "similarity_score": round(float(score) * 100, 2)
                })
            
            if len(similar_players) >= top_k:
                break

        return similar_players