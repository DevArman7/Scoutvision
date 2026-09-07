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
        raw_df = pd.read_csv(csv_path)
        raw_df.columns = [c.strip() for c in raw_df.columns]
        
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
        
        self.df = raw_df.rename(columns={k: v for k, v in column_map.items() if k in raw_df.columns}).copy()
        
        if 'id' not in self.df.columns:
            self.df['id'] = range(1, len(self.df) + 1)

        if 'position' in self.df.columns:
            self.df['position'] = self.df['position'].astype(str).str.split(',').str[0].str.strip()

        if 'league' not in self.df.columns:
            self.df['league'] = 'Top 5 European'

        if 'age' in self.df.columns:
            self.df['age'] = pd.to_numeric(self.df['age'].astype(str).str.split('-').str[0], errors='coerce').fillna(24).astype(int)
        else:
            self.df['age'] = 24

        if 'minutes' in self.df.columns:
            self.df['minutes'] = pd.to_numeric(self.df['minutes'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)
            self.df = self.df[self.df['minutes'] >= 180].reset_index(drop=True)

        self.feature_columns = []
        possible_metrics = ['goals', 'assists', 'xg', 'xa', 'progressive_passes', 'progressive_carries', 'tackles', 'interceptions']
        
        for metric in possible_metrics:
            if metric in self.df.columns:
                self.df[metric] = pd.to_numeric(self.df[metric], errors='coerce').fillna(0.0)
                p90_name = f"{metric}_p90"
                self.df[p90_name] = ((self.df[metric] / self.df['minutes']) * 90).round(2)
                self.feature_columns.append(p90_name)

        self.scaler = StandardScaler()
        self._preprocess()
        self._estimate_market_values()

    def _preprocess(self):
        for metric in self.feature_columns:
            pct_col = f"{metric}_pct"
            self.df[pct_col] = (
                self.df.groupby("position")[metric]
                .rank(pct=True) * 100
            ).round(1)

        scaled_features = self.scaler.fit_transform(self.df[self.feature_columns])
        self.similarity_matrix = cosine_similarity(scaled_features)

    def _estimate_market_values(self):
        # Algorithmic Transfer Value model:
        # Base value derived from performance percentiles + age premium + minutes weighting
        pct_cols = [f"{m}_pct" for m in self.feature_columns if f"{m}_pct" in self.df.columns]
        avg_pct = self.df[pct_cols].mean(axis=1)

        # Younger players (<24) command a strong market premium
        age_multiplier = self.df['age'].apply(lambda a: 1.8 if a <= 21 else (1.4 if a <= 24 else (1.0 if a <= 28 else 0.6)))
        min_factor = (self.df['minutes'] / 2500).clip(upper=1.2)

        # Market value in Millions (€)
        base_val = (avg_pct ** 1.35) * 0.18
        est_values = (base_val * age_multiplier * min_factor).round(1)
        self.df['estimated_value'] = est_values.clip(lower=2.0)

    def _generate_scouting_report(self, row, stats: dict) -> dict:
        def p(metric_name):
            return stats.get(metric_name, {}).get("percentile", 50.0)

        category_scores = {
            "Attacking": round((p("goals_p90") * 0.6 + p("xg_p90") * 0.4), 1),
            "Creativity": round((p("xa_p90") * 0.6 + p("assists_p90") * 0.4), 1),
            "Progression": round((p("progressive_carries_p90") * 0.5 + p("progressive_passes_p90") * 0.5), 1),
            "Passing": round(p("progressive_passes_p90"), 1),
            "Defending": round((p("tackles_p90") * 0.5 + p("interceptions_p90") * 0.5), 1)
        }

        metric_labels = {
            "goals_p90": "Goal threat",
            "xg_p90": "Box penetration & shot quality",
            "assists_p90": "Final ball execution",
            "xa_p90": "Chance creation",
            "progressive_passes_p90": "Progressive passing",
            "progressive_carries_p90": "Ball carrying & drive",
            "tackles_p90": "Defensive ground duels",
            "interceptions_p90": "Anticipation & ball recoveries"
        }

        strengths = []
        weaknesses = []

        for metric, meta in stats.items():
            pct = meta["percentile"]
            label = metric_labels.get(metric, metric.replace("_p90", "").replace("_", " ").title())
            if pct >= 75:
                strengths.append(label)
            elif pct <= 35:
                weaknesses.append(label)

        if not strengths:
            strengths.append("Balanced profile across phases")
        if not weaknesses:
            weaknesses.append("No statistical red flags")

        pos = str(row.get("position", "")).upper()
        att = category_scores["Attacking"]
        prog = category_scores["Progression"]
        dfn = category_scores["Defending"]
        cre = category_scores["Creativity"]

        if "MF" in pos:
            if prog >= 80 and att >= 75 and dfn >= 60:
                archetype = "Progressive Box-to-Box Midfielder"
            elif dfn >= 75 and prog <= 60:
                archetype = "Defensive Anchor / Ball-Winner"
            elif cre >= 80 or att >= 80:
                archetype = "Advanced Playmaker / Half-Space Creator"
            elif prog >= 75 and dfn >= 70:
                archetype = "Deep-Lying Controller"
            else:
                archetype = "Central Engine"
        elif "FW" in pos:
            if att >= 80 and cre >= 70:
                archetype = "Complete Forward"
            elif att >= 80:
                archetype = "Clinical Poacher / Pure Finisher"
            elif prog >= 75:
                archetype = "Inside Forward / Dynamic Dribbler"
            else:
                archetype = "Pressing Forward"
        elif "DF" in pos:
            if prog >= 75:
                archetype = "Ball-Playing Modern Defender"
            elif dfn >= 80:
                archetype = "Aggressive Stopper"
            else:
                archetype = "Positional Cover Defender"
        else:
            archetype = "Utility Specialist"

        return {
            "ratings": category_scores,
            "strengths": strengths[:3],
            "weaknesses": weaknesses[:3],
            "archetype": archetype
        }

    def search_players(self, position: Optional[str] = None, min_minutes: int = 0, query: Optional[str] = None) -> List[Dict[str, Any]]:
        filtered = self.df.copy()
        if position:
            filtered = filtered[filtered["position"].str.upper() == position.upper()]
        if min_minutes > 0:
            filtered = filtered[filtered["minutes"] >= min_minutes]
        if query:
            filtered = filtered[filtered["name"].str.contains(query, case=False, na=False)]
            
        columns = ["id", "name", "team", "position", "age", "minutes", "estimated_value"]
        existing = [c for c in columns if c in filtered.columns]
        return filtered[existing].head(50).to_dict(orient="records")

    def get_player_profile(self, player_id: int) -> Optional[Dict[str, Any]]:
        player_row = self.df[self.df["id"] == player_id]
        if player_row.empty:
            return None

        row = player_row.iloc[0]
        stats = {}
        for feat in self.feature_columns:
            stats[feat] = {
                "raw_value": float(row[feat]),
                "percentile": float(row.get(f"{feat}_pct", 50.0))
            }

        report = self._generate_scouting_report(row, stats)

        return {
            "id": int(row["id"]),
            "name": str(row["name"]),
            "team": str(row.get("team", "Unknown")),
            "league": str(row.get("league", "Top 5 European")),
            "position": str(row.get("position", "N/A")),
            "age": int(row.get("age", 24)),
            "minutes": int(row.get("minutes", 0)),
            "estimated_value": float(row.get("estimated_value", 10.0)),
            "stats": stats,
            "report": report
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
            if self.df.loc[match_idx, "position"] == target_pos:
                similar_players.append({
                    "id": int(self.df.loc[match_idx, "id"]),
                    "name": str(self.df.loc[match_idx, "name"]),
                    "team": str(self.df.loc[match_idx].get("team", "Unknown")),
                    "position": str(self.df.loc[match_idx, "position"]),
                    "age": int(self.df.loc[match_idx, "age"]),
                    "estimated_value": float(self.df.loc[match_idx, "estimated_value"]),
                    "similarity_score": round(float(score) * 100, 2)
                })
            if len(similar_players) >= top_k:
                break

        return similar_players

    def find_replacements(
        self,
        player_id: int,
        min_age: int = 16,
        max_age: int = 35,
        max_budget: float = 100.0,
        min_similarity: float = 70.0,
        exclude_same_team: bool = True,
        top_k: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        """Realistic recruitment targets matching budget, age window, and statistical similarity."""
        if player_id not in self.df["id"].values:
            return None

        idx = self.df.index[self.df["id"] == player_id].tolist()[0]
        target_pos = self.df.loc[idx, "position"]
        target_team = self.df.loc[idx, "team"]

        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        targets = []
        for match_idx, score in sorted_scores:
            if match_idx == idx:
                continue

            sim_pct = round(float(score) * 100, 1)
            if sim_pct < min_similarity:
                break

            m_pos = self.df.loc[match_idx, "position"]
            m_age = int(self.df.loc[match_idx, "age"])
            m_team = str(self.df.loc[match_idx, "team"])
            m_val = float(self.df.loc[match_idx, "estimated_value"])

            # Filter criteria
            if m_pos != target_pos:
                continue
            if not (min_age <= m_age <= max_age):
                continue
            if m_val > max_budget:
                continue
            if exclude_same_team and m_team == target_team:
                continue

            targets.append({
                "id": int(self.df.loc[match_idx, "id"]),
                "name": str(self.df.loc[match_idx, "name"]),
                "team": m_team,
                "position": m_pos,
                "age": m_age,
                "estimated_value": m_val,
                "similarity_score": sim_pct
            })

            if len(targets) >= top_k:
                break

        return targets