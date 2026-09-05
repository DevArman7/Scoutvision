import os
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from engine import ScoutingEngine

app = FastAPI(
    title="ScoutVision | Football Intelligence API",
    description="Statistical scouting radar and similarity recommendation engine.",
    version="1.0.0"
)

# Initialize scouting engine
scout_engine = ScoutingEngine()

# Ensure static folder exists and mount it
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
def serve_dashboard():
    """Serves the frontend dashboard."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend file 'static/index.html' not found.")
    return FileResponse(index_path)

@app.get("/api/scout/search")
def search_players(
    position: Optional[str] = Query(None, description="Filter by position: FW, MF, DF"),
    min_minutes: int = Query(0, description="Minimum minutes played"),
    q: Optional[str] = Query(None, description="Search player by name")
):
    """Search and filter players for scouting."""
    return scout_engine.search_players(position=position, min_minutes=min_minutes, query=q)

@app.get("/api/scout/player/{player_id}")
def get_player(player_id: int):
    """Retrieve player stats with raw values and position-based percentiles."""
    profile = scout_engine.get_player_profile(player_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Player not found")
    return profile

@app.get("/api/scout/player/{player_id}/similar")
def get_similar_players(
    player_id: int, 
    top_k: int = Query(5, ge=1, le=10, description="Number of matches to return")
):
    """Find the top statistically similar players using cosine similarity."""
    similar = scout_engine.get_similar_players(player_id, top_k=top_k)
    if similar is None:
        raise HTTPException(status_code=404, detail="Player not found")
    return similar