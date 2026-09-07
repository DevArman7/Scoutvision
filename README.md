# ⚽ ScoutVision

<p align="center">
  <strong>Data. Players. Future.</strong>
</p>

<p align="center">
  An ML-powered football scouting and player analytics platform designed to transform raw football statistics into meaningful scouting insights.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![HTML](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

</p>

---

# 📌 Overview

**ScoutVision** is a football analytics and scouting platform that uses player performance data and machine learning techniques to help analyze, compare, and discover football players.

Modern football scouting involves evaluating thousands of players across different positions, leagues, ages, and playing styles. Raw statistics alone can make this process difficult and time-consuming.

ScoutVision attempts to simplify this process by turning player statistics into visual and analytical insights.

The platform currently focuses on:

- Player profiling
- Performance analysis
- Percentile-based evaluation
- Statistical similarity
- Player comparison
- Player discovery
- Scouting-oriented visualization

The long-term goal is to evolve ScoutVision into a complete **football recruitment intelligence platform** capable of helping answer questions such as:

> "Who plays most like this player?"

> "Who could replace this player?"

> "Which young players are statistically similar but significantly cheaper?"

> "Which players fit a particular tactical system?"

---

# 🎯 Project Goal

The primary objective of ScoutVision is to bridge the gap between:

```text
Raw Football Data
       ↓
Data Processing
       ↓
Statistical Analysis
       ↓
Machine Learning
       ↓
Player Similarity
       ↓
Scouting Insights
       ↓
Recruitment Decisions

Instead of presenting users with a large spreadsheet of football statistics, ScoutVision provides an interactive interface where the data can be explored through player profiles, visualizations, comparisons, and similarity analysis.

✨ Features
👤 1. Player Profiles

ScoutVision provides a dedicated profile for individual players.

A player profile can contain information such as:

Player name
Position
Club
League
Age
Market value
Minutes played
Performance statistics
Per-90 statistics
Performance percentiles

Example:

Lamine Yamal

Position:
Dynamic Winger / Inside Forward

Club:
Barcelona

League:
La Liga

Age:
16

Market Value:
€82.4M

Minutes:
2,201

The profile provides a quick overview before diving deeper into the player's statistics.

📊 2. Performance Percentiles

Raw statistics do not always tell the complete story.

For example:

Player A
10 Goals

doesn't immediately tell us whether 10 goals is excellent for that player's position.

ScoutVision therefore uses percentile-based performance analysis to provide contextual information.

Example:

Goals              72nd percentile
Assists             78th percentile
xG                  76th percentile
xA                  82nd percentile
Progressive Passes  88th percentile
Progressive Carries 94th percentile

This allows a player's performance to be evaluated relative to an appropriate comparison group.

📈 3. Radar / Performance Profile

ScoutVision visualizes player performance using radar charts.

Example categories include:

                 Goals

        Progression     Assists

        Passing            xG

             xA

The radar visualization provides a quick visual representation of the player's statistical profile.

This is particularly useful when comparing players with different strengths.

🎯 4. Statistical Twins

One of the core features of ScoutVision is the Statistical Twins system.

The system attempts to find players whose statistical profiles are similar to the selected player.

Example:

Selected Player
       │
       ▼
Performance Features
       │
       ▼
Feature Processing
       │
       ▼
Similarity Calculation
       │
       ▼
Rank Similar Players
       │
       ▼
Statistical Twins

Example output:

1. Gabriel Martinelli      98.3%
2. Luis Sinisterra         97.8%
3. Edon Zhegrova           97.7%
4. Takefusa Kubo           97.7%
5. Rayan Cherki             97.1%

The purpose is not to determine who is the "better" player.

Instead, the goal is to identify players with similar statistical profiles.

🧮 5. Cosine Similarity

ScoutVision uses cosine similarity to compare player feature vectors.

A simplified representation is:

Player A
[Goals, Assists, xG, xA, Progressive Passes, ...]
                 │
                 ▼
          Feature Vector
                 │
                 ▼
          Similarity Model
                 │
                 ▼
Player B
[Goals, Assists, xG, xA, Progressive Passes, ...]

Cosine similarity can be represented mathematically as:

              A · B
Similarity = -------
             ||A|| ||B||

The result indicates how closely the two vectors point in the same direction.

A higher similarity score indicates a more similar statistical profile.

⚔️ 6. Player Comparison

ScoutVision provides a comparison workflow for evaluating two players side-by-side.

Example:

              Player A       Player B

Goals           ███████        █████████
Assists         █████████      ████████
xG              ████████       █████████
xA              █████████      █████████
Dribbling       ██████████     ███████
Progression     ██████████     ████████

This allows users to quickly identify differences in:

Goal contribution
Creativity
Progression
Passing
Chance creation
Ball carrying
Defensive contribution
🔎 7. Player Search

ScoutVision provides a player search interface.

Users can search for players by name.

Example:

Search by player name...

Goretzka
Bellingham
Bruno Fernandes
Saka
Yamal

The goal is to make player discovery fast and intuitive.

🧠 Machine Learning Pipeline

The current machine-learning workflow is centered around statistical feature analysis and player similarity.

A simplified pipeline looks like:

                Player Dataset
                     │
                     ▼
              Data Cleaning
                     │
                     ▼
             Feature Selection
                     │
                     ▼
            Feature Engineering
                     │
                     ▼
              Feature Scaling
                     │
                     ▼
             Player Vectors
                     │
                     ▼
            Similarity Analysis
                     │
                     ▼
          Ranked Similar Players
📐 Feature Engineering

Player statistics can have different scales.

For example:

Goals                  0 - 30
Assists                0 - 20
Progressive Carries    0 - 300
Passes                 0 - 3000

Directly comparing these values could cause high-volume statistics to dominate the similarity calculation.

Therefore, feature preprocessing and normalization are important parts of the machine-learning pipeline.

The general idea is:

Raw Statistics
      ↓
Clean Data
      ↓
Select Relevant Features
      ↓
Normalize / Scale
      ↓
Generate Feature Vectors
      ↓
Calculate Similarity
🧩 Example Player Feature Vector

A simplified player representation could look like:

[
    goals,
    assists,
    xG,
    xA,
    progressive_passes,
    progressive_carries,
    dribbles,
    tackles,
    interceptions
]

These features can then be processed and used for similarity calculations.

🏗️ Architecture

The overall ScoutVision architecture can be represented as:

                         ┌───────────────────┐
                         │       User        │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │   ScoutVision UI  │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │  Python Backend   │
                         └─────────┬─────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
             Player Data      Analytics       ML Engine
                    │              │              │
                    └──────────────┼──────────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Scouting Insights │
                         └───────────────────┘
🛠️ Technology Stack
Backend / Data Science
Python

Python is used as the primary programming language for the analytics and machine-learning components.

Pandas

Used for:

Dataset loading
Data cleaning
Data transformation
Statistical analysis
NumPy

Used for:

Numerical operations
Vector calculations
Mathematical processing
Scikit-learn

Used for machine-learning and similarity-related operations.

Potential applications include:

Feature preprocessing
Scaling
Similarity calculations
Clustering
Predictive modelling
🎨 Frontend

ScoutVision uses a custom web-based interface built using:

HTML5
CSS3
JavaScript

The UI follows a dark analytics-oriented design intended to resemble modern sports analytics platforms.

Key UI components include:

Player cards
Search interface
Radar charts
Statistical comparison cards
Similarity rankings
Performance metrics
Recruitment controls
📂 Project Structure

The exact structure may evolve as the project grows, but the intended architecture is:

ScoutVision/
│
├── data/
│   ├── players.csv
│   └── processed_players.csv
│
├── models/
│   ├── similarity_model.py
│   ├── clustering_model.py
│   └── prediction_model.py
│
├── services/
│   ├── player_service.py
│   ├── scouting_service.py
│   ├── similarity_service.py
│   └── recommendation_service.py
│
├── templates/
│   ├── index.html
│   ├── player.html
│   ├── compare.html
│   └── scouting.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   ├── js/
│   │   └── app.js
│   │
│   └── images/
│
├── notebooks/
│   ├── data_analysis.ipynb
│   └── model_experiments.ipynb
│
├── tests/
│   ├── test_players.py
│   └── test_similarity.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md

The actual implementation may differ from this structure depending on the current development stage.

💻 Installation
1. Clone the Repository
git clone https://github.com/YOUR_USERNAME/ScoutVision.git

Move into the project directory:

cd ScoutVision
🐍 2. Create a Virtual Environment
Windows
python -m venv venv

Activate the environment:

venv\Scripts\activate
Linux / macOS
python3 -m venv venv

Activate:

source venv/bin/activate
📦 3. Install Dependencies

Install the required Python packages:

pip install -r requirements.txt

If a requirements.txt file has not yet been created, install the primary dependencies:

pip install pandas numpy scikit-learn

Additional dependencies may be required depending on the backend framework used by the project.

▶️ 4. Run ScoutVision

Run the application using the project's entry point.

For example:

python app.py

If the application uses a web framework, open the local development address provided by the server.

For example:

http://127.0.0.1:5000

or:

http://127.0.0.1:8000
📊 Dataset

ScoutVision currently focuses on football player data from the Top 5 European leagues.

The dataset can contain information such as:

Player
Club
League
Position
Age
Minutes
Goals
Assists
xG
xA
Progressive Passes
Progressive Carries
Dribbles
Tackles
Interceptions

The exact columns depend on the dataset being used by the current implementation.

⚠️ Data Disclaimer

ScoutVision is an educational and analytical project.

The football data used by the application may come from publicly available datasets or third-party football data sources.

ScoutVision does not claim ownership of third-party data.

Player values, statistics, club information, and other football-related information may change over time and should not be treated as official transfer or scouting information.

ScoutVision is not affiliated with:

FC Barcelona
Bayern Munich
Real Madrid
Premier League
La Liga
Bundesliga
Serie A
Ligue 1
UEFA
FIFA
Any individual football player

unless explicitly stated otherwise.

🔬 Machine Learning Methodology

The core idea behind ScoutVision is to represent every player as a numerical feature vector.

For example:

Player A

[0.24, 0.34, 0.28, 0.37, 7.3, 5.8, 6.1, ...]

Another player can be represented similarly:

Player B

[0.31, 0.29, 0.32, 0.35, 7.8, 5.2, 5.9, ...]

The vectors can then be compared mathematically.

The general workflow is:

             Player Statistics
                    │
                    ▼
             Data Cleaning
                    │
                    ▼
            Feature Selection
                    │
                    ▼
              Normalization
                    │
                    ▼
            Feature Vectors
                    │
                    ▼
          Cosine Similarity
                    │
                    ▼
             Similarity Score
                    │
                    ▼
            Ranked Results
🎯 Why Similarity?

Traditional player scouting often requires manually watching and comparing many players.

A statistical similarity system can reduce the search space.

For example:

Scout wants:
"Find players similar to Player X."

             ↓

ScoutVision
             ↓

Analyze Player X
             ↓

Compare against dataset
             ↓

Calculate similarity
             ↓

Rank candidates
             ↓

Top 5 / Top 10 players

This does not replace human scouting.

Instead, it can act as a data-driven scouting assistant.

🔮 Planned Features

ScoutVision is designed to grow beyond basic player similarity.

The following features are planned or under consideration.

🧠 1. Role-Aware Similarity

Instead of comparing every player using the same features, players will be compared according to their tactical role.

For example:

Midfield Roles
Box-to-Box Midfielder
Deep-Lying Playmaker
Advanced Playmaker
Ball-Winning Midfielder
Mezzala
Forward Roles
Poacher
Target Man
Inside Forward
Pressing Forward
Complete Forward
Defensive Roles
Ball-Playing Defender
Stopper
Full Back
Wing Back
Inverted Full Back

This would make player similarity more meaningful.

🔎 2. Find a Replacement

A major planned feature is a recruitment-oriented replacement engine.

Example:

Find replacement for:

Lamine Yamal

Position:
RW

Age:
18 - 23

Maximum Market Value:
€50M

Minimum Similarity:
80%

ScoutVision could then return:

Player A     94.2%
Player B     91.8%
Player C     89.6%
Player D     87.3%

The purpose is to transform statistical similarity into a practical recruitment workflow.

🎯 3. Tactical Fit Score

A future version of ScoutVision will evaluate whether a player's profile fits a particular playing style.

For example:

Possession         █████████░
High Press         ████████░░
Build-up Play      █████████░
Counter Attack     █████░░░░░
Direct Play        ████░░░░░░

The system could then calculate:

Tactical Fit

Attacking       94%
Creativity      96%
Progression     91%
Pressing        84%
Possession      93%

Overall Fit     92%
💎 4. Hidden Gem Finder

One of the planned recruitment features is an undervalued player discovery system.

The basic concept:

High Performance
       +
Young Age
       +
Low Market Value
       +
Strong Potential
       ↓
   HIDDEN GEM

Example:

Player       Age     Value      Score

Player A      20      €8M        91
Player B      21      €12M       89
Player C      19      €5M        88

This could help identify players who provide strong statistical value relative to their market valuation.

📋 5. Scouting Shortlists

Users will eventually be able to save interesting players.

Example:

MY SHORTLIST

Right Wingers

1. Player A      91% Fit
2. Player B      88% Fit
3. Player C      86% Fit

Possible future capabilities:

Add/remove players
Organize players by position
Add scouting notes
Compare shortlisted players
Export scouting reports
📊 6. Player Clustering

ScoutVision may use unsupervised machine learning to group players according to statistical characteristics.

Potential algorithms:

K-Means
Hierarchical Clustering
PCA

Example:

                  ● ●
               ● ● ●
                         ●
                      ● ● ●

        ● ●
      ● ● ●

                     ● ●

Clusters could represent groups such as:

Cluster 1
Creative Midfielders

Cluster 2
Ball-Winning Midfielders

Cluster 3
Progressive Wingers

Cluster 4
Defensive Fullbacks

This can help discover player archetypes without manually defining every category.

📈 7. Player Development Prediction

A future predictive model could estimate how a player's statistical profile might develop over time.

Example:

Performance Score

2023/24     78
2024/25     84
2025/26     88
2026/27     91  ← Projection

Potential input variables:

Age
Minutes
Historical performance
Position
Playing time
Previous development trajectory
💰 8. Transfer Budget Optimizer

Another planned feature is recruitment optimization based on a transfer budget.

Example:

Transfer Budget: €100M

Required Positions:

RW
CB
DM

ScoutVision could search for combinations such as:

Option A

RW Player A     €30M
CB Player B     €40M
DM Player C     €25M

Total           €95M

The system could rank recruitment combinations based on:

Player quality
Similarity
Tactical fit
Age
Market value
Squad requirements
🏆 9. Squad Analysis

A future Squad Builder will allow users to create a squad and analyze its overall profile.

Example:

Formation: 4-3-3

             ST

       LW          RW

          CM    CM

             DM

LB      CB      CB      RB

             GK

ScoutVision could then calculate:

Attack       91
Midfield     87
Defense      82
Depth        74
Age Profile  81

The system could also identify weaknesses.

Example:

⚠ Squad Weakness Detected

Right-back depth is below squad average.

Recommended players:
Player A
Player B
Player C
🤖 10. Natural Language AI Scout

A long-term goal is to add a natural-language scouting assistant.

Instead of manually configuring filters, a user could write:

Find me a young box-to-box midfielder
under €30M who is good at progressing
the ball and fits a high-pressing team.

The system could convert the request into structured constraints:

Position = CM
Role = Box-to-Box
Age < 24
Market Value < €30M
High Progression
High Pressing Fit

Then ScoutVision could return ranked candidates.

🧠 Explainable Recommendations

A major goal is to make recommendations explainable.

Instead of simply showing:

Similarity = 94.2%

ScoutVision should explain:

WHY THIS PLAYER?

Progressive Carries      96%
Progressive Passes       91%
xA                        89%
Dribbling                 95%
Goal Contribution         78%

Overall Similarity        94.2%

This allows users to understand why the model considers two players similar.

🗺️ Development Roadmap

ScoutVision is being developed in multiple stages.

🟢 Phase 1 — Player Analytics
 Player search
 Player profiles
 Performance statistics
 Percentile visualization
 Radar chart
 Statistical Twins
 Similarity analysis
🔵 Phase 2 — Scouting Intelligence
 Role-based similarity
 Advanced player filters
 Player comparison
 Find-a-Replacement engine
 Scouting shortlist
 Automated scouting reports
🟣 Phase 3 — Recruitment Analytics
 Tactical Fit Score
 Hidden Gem Finder
 Transfer Budget Optimizer
 Squad Builder
 Squad Weakness Detection
 League Comparison
🔴 Phase 4 — Advanced Machine Learning
 Player clustering
 Player role classification
 Development prediction
 Recommendation ranking
 Explainable ML
 Multi-league scouting
🟡 Phase 5 — AI Scout
 Natural-language player search
 AI scouting assistant
 AI-generated scouting reports
 Conversational recruitment analysis
 Context-aware recommendations
🧪 Example Use Case

Imagine a club needs to replace an attacking midfielder.

The traditional workflow might look like:

Watch matches
      ↓
Search databases
      ↓
Compare statistics
      ↓
Create shortlist
      ↓
Scout players

ScoutVision aims to simplify the data-analysis portion:

                Recruitment Need
                       │
                       ▼
               Define Requirements
                       │
                       ▼
                ScoutVision
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
     Player Data              ML Analysis
          │                         │
          └────────────┬────────────┘
                       ▼
              Candidate Ranking
                       │
                       ▼
                 Shortlist
                       │
                       ▼
              Human Evaluation

The system is intended to assist scouts rather than replace professional scouting judgment.

🧑‍💻 Development Philosophy

ScoutVision follows three core principles.

1. Data

Football decisions should be supported by measurable evidence.

2. Players

Every player has a unique statistical and tactical profile.

3. Future

Machine learning can help discover talent and recruitment opportunities that may otherwise be difficult to identify.

Hence:

Data. Players. Future.

📚 What This Project Demonstrates

ScoutVision combines several areas of software and data science.

Python Development
Data processing
Backend logic
Application architecture
Automation
Data Science
Data cleaning
Feature engineering
Statistical analysis
Data visualization
Machine Learning
Feature scaling
Similarity analysis
Clustering
Recommendation systems
Predictive modelling
Frontend Development
Responsive UI
Interactive dashboards
Data visualization
User interaction
Software Engineering
Modular architecture
Separation of concerns
Testing
Version control
Documentation
🔐 Future Production Architecture

As ScoutVision grows, the architecture can evolve into:

                         ┌────────────────────┐
                         │      Frontend      │
                         │ HTML/CSS/JS/React  │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │       REST API     │
                         └─────────┬──────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
       Player Service       ML Service          Scouting Service
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │     PostgreSQL     │
                         └────────────────────┘

Potential production technologies may include:

Frontend:
React / Next.js

Backend:
Python / FastAPI

Database:
PostgreSQL

ML:
Scikit-learn / PyTorch

Deployment:
Docker
Cloud Platform

AI:
LLM + Retrieval / Tool-based Analytics

These technologies are part of the future architecture and are not necessarily part of the current implementation.

🧪 Testing

Testing is an important part of making ScoutVision reliable.

Future test coverage will include:

Player Search
      ↓
Player Data Validation
      ↓
Feature Processing
      ↓
Similarity Calculation
      ↓
Recommendation Logic

Example:

def test_similarity():
    similarity = calculate_similarity(player_a, player_b)

    assert 0 <= similarity <= 1
📌 Known Limitations

The current version of ScoutVision has several limitations.

Dataset limitations

The quality of the recommendations depends heavily on the quality and completeness of the underlying dataset.

Statistical similarity limitations

Two players can have similar statistics while playing very different tactical roles.

For example:

Similar Statistics
        ≠
Same Playing Style

This is one of the reasons role-aware similarity and tactical analysis are planned for future versions.

Market value limitations

Market values are not always an accurate representation of a player's actual transfer cost.

Human scouting remains important

Statistics cannot completely capture:

Decision making
Personality
Tactical intelligence
Communication
Adaptability
Dressing-room influence
Injury context
Match-specific behavior

ScoutVision is therefore designed as a scouting support system, not a replacement for human scouts.

🚀 Future Vision

The ultimate vision for ScoutVision is to move from:

PLAYER ANALYTICS

to:

PLAYER INTELLIGENCE

and eventually:

RECRUITMENT INTELLIGENCE

The long-term system could answer questions such as:

"Who are the best young replacements for this player?"

"Which players are undervalued relative to their performance?"

"Which midfielder best fits this team's tactical system?"

"Which players have similar development trajectories?"

"Build me a shortlist under a €50M budget."

The goal is to make football recruitment more data-driven, explainable, and efficient.

🤝 Contributing

Contributions, suggestions, and ideas are welcome.

Fork the Repository

Create your own fork of the project.

Create a Feature Branch
git checkout -b feature/new-feature
Make Your Changes

Implement and test your feature.

Commit
git add .
git commit -m "Add new scouting feature"
Push
git push origin feature/new-feature

Then open a Pull Request.

⭐ Support

If you find ScoutVision interesting or useful, consider giving the repository a ⭐.

It helps support the continued development of the project.

📬 Feedback

Suggestions and feedback are welcome.

If you find a bug or have an idea for a new feature, feel free to open an Issue.

📜 License

This project is intended primarily for educational, research, and portfolio purposes.

See the LICENSE file for the complete license information.

👨‍💻 Author
SK . Arman Aman

B.Tech Engineering Student

Interested in:

Software Development
Java
Python
Machine Learning
Data Science
Backend Development
System Design

ScoutVision is an ongoing project combining these interests with football analytics.

⚽ ScoutVision
<p align="center">
Data. Players. Future.

Turning football data into scouting intelligence.

</p> ```
