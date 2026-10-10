from typing import Dict, Any, Optional

def get_player_image_metadata(player_name: str, team_name: str = "") -> Dict[str, Optional[str]]:
    """
    Returns image metadata dictionary for a player.
    To avoid cross-contamination and portrait mismatch bugs (e.g. Jude Bellingham
    receiving Yudai Baba's SofaScore ID 993510), unverified external IDs are never
    fabricated or assumed interchangeable.
    """
    clean_name = player_name.strip() if player_name else ""
    return {
        "name": clean_name,
        "sofascore_id": None,
        "fotmob_id": None,
        "image_url": None
    }

