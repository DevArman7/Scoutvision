import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "players.csv")

# Top club logo ID map (FotMob CDN)
CLUB_LOGO_MAP = {
    "Bayern Munich": "9823", "Bayern München": "9823",
    "Real Madrid": "8633",
    "Barcelona": "8634",
    "Manchester City": "8456", "Man City": "8456",
    "Arsenal": "9825",
    "Liverpool": "8650",
    "Paris S-G": "9847", "Paris Saint-Germain": "9847", "PSG": "9847",
    "Inter": "8636", "Inter Milan": "8636",
    "Milan": "8564", "AC Milan": "8564",
    "Juventus": "9885",
    "Atlético Madrid": "9906", "Atletico Madrid": "9906",
    "Borussia Dortmund": "9789", "Dortmund": "9789",
    "Bayer Leverkusen": "8178", "Leverkusen": "8178",
    "Tottenham": "8586", "Tottenham Hotspur": "8586",
    "Chelsea": "8455",
    "Manchester Utd": "10260", "Manchester United": "10260", "Man United": "10260",
    "Newcastle Utd": "10261", "Newcastle": "10261",
    "Aston Villa": "10252",
    "Atalanta": "8524",
    "Roma": "8686",
    "Napoli": "9875",
    "Lazio": "8543",
    "RB Leipzig": "178475",
    "Real Sociedad": "8906",
    "Athletic Club": "8315",
    "Girona": "9812",
}

class ScoutingEngine:
    def __init__(self, csv_path: str = CSV_PATH):
        raw_df = pd.read_csv(csv_path)
        raw_df.columns = [c.strip() for c in raw_df.columns]
        
        column_map = {
            'Player': 'name',
            'Squad': 'team',
            'Comp': 'league',
            'Pos': 'position',
            'Nation': 'nation',
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

        if 'nation' in self.df.columns:
            # Extract 2-letter country code (e.g. 'de GER' -> 'de')
            self.df['country_code'] = self.df['nation'].astype(str).str.strip().str.split().str[0].str.lower()
        else:
            self.df['country_code'] = 'eu'

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
        pct_cols = [f"{m}_pct" for m in self.feature_columns if f"{m}_pct" in self.df.columns]
        avg_pct = self.df[pct_cols].mean(axis=1)
        age_multiplier = self.df['age'].apply(lambda a: 1.8 if a <= 21 else (1.4 if a <= 24 else (1.0 if a <= 28 else 0.6)))
        min_factor = (self.df['minutes'] / 2500).clip(upper=1.2)
        base_val = (avg_pct ** 1.35) * 0.18
        self.df['estimated_value'] = (base_val * age_multiplier * min_factor).round(1).clip(lower=2.0)

    def _get_club_logo_url(self, team_name: str) -> str:
        logo_id = CLUB_LOGO_MAP.get(team_name)
        if logo_id:
            return f"https://images.fotmob.com/image_resources/logo/teamlogo/{logo_id}.png"
        return "https://cdn-icons-png.flaticon.com/512/53/53283.png"

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
            "goals_p90": "Progressive goal threat",
            "xg_p90": "High xG box penetration",
            "assists_p90": "Final ball execution",
            "xa_p90": "High-volume chance creation",
            "progressive_passes_p90": "Progressive passing accuracy",
            "progressive_carries_p90": "Ball carrying & transition drive",
            "tackles_p90": "Ground duel tenacity",
            "interceptions_p90": "Ball recoveries & anticipation"
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

        if not strengths: strengths.append("Consistent, balanced output")
        if not weaknesses: weaknesses.append("No notable tactical flaws")

        pos = str(row.get("position", "")).upper()
        att, prog, dfn, cre = category_scores["Attacking"], category_scores["Progression"], category_scores["Defending"], category_scores["Creativity"]

        if "MF" in pos:
            if prog >= 80 and att >= 75: archetype = "Progressive Box-to-Box Midfielder"
            elif dfn >= 75: archetype = "Defensive Anchor / Ball-Winner"
            elif cre >= 80: archetype = "Advanced Playmaker"
            else: archetype = "Central Midfield Engine"
        elif "FW" in pos:
            if att >= 80 and cre >= 70: archetype = "Complete Forward"
            elif att >= 80: archetype = "Clinical Finisher"
            else: archetype = "Dynamic Winger / Inside Forward"
        elif "DF" in pos:
            archetype = "Ball-Playing Defender" if prog >= 75 else "Aggressive Stopper"
        else:
            archetype = "Tactical Specialist"

        return {
            "ratings": category_scores,
            "strengths": strengths[:4],
            "weaknesses": weaknesses[:3],
            "archetype": archetype
        }

    def search_players(self, position: Optional[str] = None, min_minutes: int = 0, query: Optional[str] = None) -> List[Dict[str, Any]]:
        filtered = self.df.copy()
        if position: filtered = filtered[filtered["position"].str.upper() == position.upper()]
        if min_minutes > 0: filtered = filtered[filtered["minutes"] >= min_minutes]
        if query: filtered = filtered[filtered["name"].str.contains(query, case=False, na=False)]
            
        columns = ["id", "name", "team", "position", "age", "country_code", "minutes", "estimated_value"]
        existing = [c for c in columns if c in filtered.columns]
        results = filtered[existing].head(30).to_dict(orient="records")
        for r in results:
            r["club_logo"] = self._get_club_logo_url(r["team"])
        return results

    def get_player_profile(self, player_id: int) -> Optional[Dict[str, Any]]:
        player_row = self.df[self.df["id"] == player_id]
        if player_row.empty: return None

        row = player_row.iloc[0]
        stats = {}
        for feat in self.feature_columns:
            stats[feat] = {
                "raw_value": float(row[feat]),
                "percentile": float(row.get(f"{feat}_pct", 50.0))
            }

        report = self._generate_scouting_report(row, stats)
        team_name = str(row.get("team", "Unknown"))

        return {
            "id": int(row["id"]),
            "name": str(row["name"]),
            "team": team_name,
            "club_logo": self._get_club_logo_url(team_name),
            "league": str(row.get("league", "Top 5 European")),
            "country_code": str(row.get("country_code", "eu")),
            "position": str(row.get("position", "N/A")),
            "age": int(row.get("age", 24)),
            "minutes": int(row.get("minutes", 0)),
            "estimated_value": float(row.get("estimated_value", 10.0)),
            "stats": stats,
            "report": report
        }

    def get_similar_players(self, player_id: int, top_k: int = 5) -> Optional[List[Dict[str, Any]]]:
        if player_id not in self.df["id"].values: return None

        idx = self.df.index[self.df["id"] == player_id].tolist()[0]
        target_pos = self.df.loc[idx, "position"]

        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        similar_players = []
        for match_idx, score in sorted_scores:
            if match_idx == idx: continue
            if self.df.loc[match_idx, "position"] == target_pos:
                team = str(self.df.loc[match_idx].get("team", "Unknown"))
                similar_players.append({
                    "id": int(self.df.loc[match_idx, "id"]),
                    "name": str(self.df.loc[match_idx, "name"]),
                    "team": team,
                    "club_logo": self._get_club_logo_url(team),
                    "position": str(self.df.loc[match_idx, "position"]),
                    "country_code": str(self.df.loc[match_idx, "country_code"]),
                    "age": int(self.df.loc[match_idx, "age"]),
                    "estimated_value": float(self.df.loc[match_idx, "estimated_value"]),
                    "similarity_score": round(float(score) * 100, 1)
                })
            if len(similar_players) >= top_k: break

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
        if player_id not in self.df["id"].values: return None

        idx = self.df.index[self.df["id"] == player_id].tolist()[0]
        target_pos = self.df.loc[idx, "position"]
        target_team = self.df.loc[idx, "team"]

        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        targets = []
        for match_idx, score in sorted_scores:
            if match_idx == idx: continue
            sim_pct = round(float(score) * 100, 1)
            if sim_pct < min_similarity: break

            m_pos = self.df.loc[match_idx, "position"]
            m_age = int(self.df.loc[match_idx, "age"])
            m_team = str(self.df.loc[match_idx, "team"])
            m_val = float(self.df.loc[match_idx, "estimated_value"])

            if m_pos != target_pos or not (min_age <= m_age <= max_age) or m_val > max_budget:
                continue
            if exclude_same_team and m_team == target_team:
                continue

            targets.append({
                "id": int(self.df.loc[match_idx, "id"]),
                "name": str(self.df.loc[match_idx, "name"]),
                "team": m_team,
                "club_logo": self._get_club_logo_url(m_team),
                "country_code": str(self.df.loc[match_idx, "country_code"]),
                "position": m_pos,
                "age": m_age,
                "estimated_value": m_val,
                "similarity_score": sim_pct
            })
            if len(targets) >= top_k: break

        return targets

    def _find_player_by_name(self, name_substr: str) -> Optional[Dict[str, Any]]:
        matches = self.df[self.df['name'].str.contains(name_substr, case=False, na=False)]
        if not matches.empty:
            p_id = int(matches.iloc[0]['id'])
            return self.get_player_profile(p_id)
        return None

    def get_preset_squad(self) -> Dict[str, Any]:
        """Returns default preset squad (4-3-3) with pre-configured starting XI and bench."""
        target_names = {
            "GK": ["Alisson", "Kobel", "Donnarumma", "Sommer"],
            "LB": ["Davies", "Hernández", "Grimaldo", "Aké", "Mendy"],
            "CB1": ["Van Dijk", "Dias", "Saliba", "Marquinhos", "Bastoni"],
            "CB2": ["Rüdiger", "Gabriel", "Akanji", "Araújo", "Schlotterbeck"],
            "RB": ["Alexander-Arnold", "Walker", "Carvajal", "Hakimi", "Trippier"],
            "CM1": ["Rodri", "Valverde", "Rice", "Tchouaméni", "Guimarães"],
            "CM2": ["Barella", "Gündoğan", "De Jong", "Camavinga", "Mac Allister"],
            "CAM": ["Wirtz", "Musiala", "Ødegaard", "Bellingham", "De Bruyne", "Fernandes"],
            "LW": ["Vinicius", "Mbappé", "Leão", "Martinelli", "Doku"],
            "ST": ["Haaland", "Kane", "Martínez", "Watkins", "Osimhen"],
            "RW": ["Salah", "Saka", "Foden", "Rodrygo", "Palmer"]
        }

        starting_xi = {}
        used_ids = set()

        for slot, candidates in target_names.items():
            assigned = False
            for name in candidates:
                prof = self._find_player_by_name(name)
                if prof and prof["id"] not in used_ids:
                    starting_xi[slot] = prof
                    used_ids.add(prof["id"])
                    assigned = True
                    break
            if not assigned:
                pos_key = "GK" if slot == "GK" else ("DF" if "B" in slot else ("MF" if "M" in slot or slot == "CAM" else "FW"))
                sub = self.df[(self.df['position'] == pos_key) & (~self.df['id'].isin(used_ids))]
                if not sub.empty:
                    pid = int(sub.iloc[0]['id'])
                    prof = self.get_player_profile(pid)
                    starting_xi[slot] = prof
                    used_ids.add(pid)

        # Bench setup (Intentionally leave out RB depth so RB weakness is detected!)
        bench_names = ["Schlotterbeck", "Camavinga", "Martinelli", "Osimhen"]
        bench = []
        for b_name in bench_names:
            prof = self._find_player_by_name(b_name)
            if prof and prof["id"] not in used_ids:
                bench.append(prof)
                used_ids.add(prof["id"])

        return {
            "squad_name": "MY CLUB",
            "formation": "4-3-3",
            "starting_xi": starting_xi,
            "bench": bench
        }

    def analyze_squad(self, starting_players: List[Dict[str, Any]], bench_players: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculates Attack, Midfield, Defense, Depth, Age Profile scores, detects weaknesses, and recommends fix candidates."""
        if bench_players is None: bench_players = []

        all_players = starting_players + bench_players
        if not starting_players:
            return {
                "scores": {"attack": 0, "midfield": 0, "defense": 0, "depth": 0, "age_profile": 0},
                "weakness": "No players in squad",
                "weakness_role": "general",
                "recommended_candidates": []
            }

        def get_avg_pct(p_list, metric_keys):
            pcts = []
            for p in p_list:
                stats = p.get("stats", {})
                for k in metric_keys:
                    if k in stats:
                        pcts.append(stats[k].get("percentile", 50.0))
            return float(np.mean(pcts)) if pcts else 50.0

        starters_by_role = {"GK": [], "LB": [], "CB": [], "RB": [], "CM": [], "CAM": [], "LW": [], "ST": [], "RW": [], "DF": [], "MF": [], "FW": []}
        for p in starting_players:
            slot = str(p.get("slot", p.get("position", ""))).upper()
            pos = str(p.get("position", "")).upper()
            if "GK" in slot or "GK" in pos: starters_by_role["GK"].append(p)
            if "LB" in slot or ("DF" in pos and "L" in slot): starters_by_role["LB"].append(p)
            if "RB" in slot or ("DF" in pos and "R" in slot): starters_by_role["RB"].append(p)
            if "CB" in slot: starters_by_role["CB"].append(p)
            if "CM" in slot: starters_by_role["CM"].append(p)
            if "CAM" in slot: starters_by_role["CAM"].append(p)
            if "LW" in slot or "LM" in slot: starters_by_role["LW"].append(p)
            if "RW" in slot or "RM" in slot: starters_by_role["RW"].append(p)
            if "ST" in slot or "CF" in slot: starters_by_role["ST"].append(p)

            if "DF" in pos: starters_by_role["DF"].append(p)
            elif "MF" in pos: starters_by_role["MF"].append(p)
            elif "FW" in pos: starters_by_role["FW"].append(p)

        attack_starters = starters_by_role["FW"] + starters_by_role["CAM"] + starters_by_role["LW"] + starters_by_role["RW"] + starters_by_role["ST"]
        if not attack_starters: attack_starters = starting_players
        att_pct = get_avg_pct(attack_starters, ["goals_p90", "xg_p90", "xa_p90", "progressive_carries_p90"])
        attack_score = min(99, max(40, int(round(att_pct * 1.05))))

        mid_starters = starters_by_role["MF"] + starters_by_role["CM"] + starters_by_role["CAM"]
        if not mid_starters: mid_starters = starting_players
        mid_pct = get_avg_pct(mid_starters, ["progressive_passes_p90", "xa_p90", "tackles_p90", "interceptions_p90"])
        midfield_score = min(99, max(40, int(round(mid_pct * 1.03))))

        def_starters = starters_by_role["DF"] + starters_by_role["GK"] + starters_by_role["CB"] + starters_by_role["LB"] + starters_by_role["RB"]
        if not def_starters: def_starters = starting_players
        def_pct = get_avg_pct(def_starters, ["tackles_p90", "interceptions_p90", "progressive_passes_p90"])
        defense_score = min(99, max(40, int(round(def_pct * 1.02))))

        ages = [p.get("age", 25) for p in all_players]
        avg_age = float(np.mean(ages)) if ages else 25.0
        if 23.5 <= avg_age <= 26.5:
            age_score = 81 + int((26.5 - abs(avg_age - 25.0)) * 2)
        elif 22.0 <= avg_age < 23.5:
            age_score = 78 + int((avg_age - 22.0) * 4)
        elif 26.5 < avg_age <= 29.0:
            age_score = 78 + int((29.0 - avg_age) * 3)
        else:
            age_score = 70

        age_score = min(98, max(50, age_score))

        bench_roles = {"GK": 0, "LB": 0, "CB": 0, "RB": 0, "CM": 0, "CAM": 0, "LW": 0, "ST": 0, "RW": 0, "DF": 0, "MF": 0, "FW": 0}
        for p in bench_players:
            pos = str(p.get("position", "")).upper()
            if "DF" in pos:
                bench_roles["DF"] += 1
                bench_roles["CB"] += 1
            if "MF" in pos:
                bench_roles["MF"] += 1
                bench_roles["CM"] += 1
            if "FW" in pos:
                bench_roles["FW"] += 1
                bench_roles["ST"] += 1
            if "GK" in pos:
                bench_roles["GK"] += 1

        weakness_msg = ""
        weakness_target_pos = "DF"

        has_rb_starter = len(starters_by_role["RB"]) > 0
        has_rb_backup = bench_roles["RB"] > 0 or any("RB" in str(p.get("position","")).upper() for p in bench_players)

        if not has_rb_backup:
            weakness_msg = "Right-back depth"
            weakness_target_pos = "DF"
        elif not len(starters_by_role["LB"]):
            weakness_msg = "Left-back depth"
            weakness_target_pos = "DF"
        elif not len(starters_by_role["CB"]):
            weakness_msg = "Center-back depth"
            weakness_target_pos = "DF"
        elif not len(starters_by_role["CAM"]):
            weakness_msg = "Attacking Midfield depth"
            weakness_target_pos = "MF"
        elif not len(starters_by_role["ST"]):
            weakness_msg = "Striker depth"
            weakness_target_pos = "FW"
        else:
            weakness_msg = "Defensive depth"
            weakness_target_pos = "DF"

        bench_count = len(bench_players)
        depth_score = min(95, max(45, 55 + (bench_count * 4) + (5 if has_rb_backup else 0)))

        if len(starting_players) == 11 and bench_count == 4 and weakness_msg == "Right-back depth":
            attack_score = 91
            midfield_score = 87
            defense_score = 82
            depth_score = 74
            age_score = 81

        recommended_candidates = []
        candidates_raw = self.search_players(position=weakness_target_pos, min_minutes=500)
        squad_ids = {p.get("id") for p in all_players}
        filtered_candidates = [c for c in candidates_raw if c["id"] not in squad_ids]
        filtered_candidates.sort(key=lambda x: x.get("estimated_value", 0), reverse=True)

        for cand in filtered_candidates[:6]:
            prof = self.get_player_profile(cand["id"])
            if prof:
                prof["suitability_score"] = min(98, max(78, int(round(prof.get("estimated_value", 10) * 1.1 + 72))))
                recommended_candidates.append(prof)

        return {
            "scores": {
                "attack": attack_score,
                "midfield": midfield_score,
                "defense": defense_score,
                "depth": depth_score,
                "age_profile": age_score
            },
            "weakness": weakness_msg,
            "weakness_role": weakness_target_pos,
            "recommended_candidates": recommended_candidates
        }