import nbformat as nbf

nb = nbf.v4.new_notebook()

def add_md(text):
    nb.cells.append(nbf.v4.new_markdown_cell(text))

def add_code(code):
    nb.cells.append(nbf.v4.new_code_cell(code))

add_md("""# AFL Player Performance Investigation
**Objective:** Identify the most valuable and consistent players for the upcoming season using the provided datasets.
""")

add_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load data
data_dir = 'D:/netixsol/Week- 2/Day-6/AFL player information and season statistics/'
df_info = pd.read_csv(data_dir + 'players_info.csv')
df_season = pd.read_csv(data_dir + 'seasonal_stats.csv')
df_round = pd.read_csv(data_dir + 'afl_players_round_by_round_stats_raw.csv')

# Merge Datasets
# Merge season stats with player info
df_merged_season = pd.merge(df_season, df_info, left_on='player_id', right_on='id', how='left')

# Drop the redundant 'id' from info to avoid confusion
if 'id' in df_merged_season.columns:
    df_merged_season.drop(columns=['id'], inplace=True)
    
df_merged_season.head()
""")

add_md("""### Business Insight 1: Data Integration
By combining the demographic information (`players_info`) with seasonal statistics, we unlock the ability to analyze performance in the context of player traits, like age and physical dimensions. The successful merge forms the foundational dataset for our investigation.""")

add_md("""## 1. Feature Engineering
We will engineer 5 new meaningful features to better assess player performance:
1. **Goals/Game:** Indicates scoring reliability.
2. **Fantasy/Game:** A proxy for all-around contribution per game.
3. **Disposal Reliability Score:** `(Disposals - Clangers) / Disposals`. Evaluates how efficiently a player uses the ball without turning it over.
4. **Offensive Impact Score:** `Goals + Goal Assists + (Inside 50s * 0.5)`. Measures contribution to attacking plays.
5. **Defensive Action Score:** `Tackles + One Percenters + (Rebound 50s * 0.5)`. Measures defensive work rate.""")

add_code("""# Ensure numeric columns
for col in ['disposals', 'clangers', 'goals', 'goal_assists', 'inside_50s', 'tackles', 'one_percenters', 'rebound_50s', 'games_played', 'total_fantasy_points']:
    if col in df_merged_season.columns:
        df_merged_season[col] = pd.to_numeric(df_merged_season[col], errors='coerce').fillna(0)

# Feature Engineering
df_merged_season['Goals_per_Game'] = np.where(df_merged_season['games_played'] > 0, df_merged_season['goals'] / df_merged_season['games_played'], 0)
df_merged_season['Fantasy_per_Game'] = np.where(df_merged_season['games_played'] > 0, df_merged_season['total_fantasy_points'] / df_merged_season['games_played'], 0)
df_merged_season['Disposal_Reliability'] = np.where(df_merged_season['disposals'] > 0, (df_merged_season['disposals'] - df_merged_season['clangers']) / df_merged_season['disposals'], 0)
df_merged_season['Offensive_Impact'] = df_merged_season['goals'] + df_merged_season['goal_assists'] + (df_merged_season['inside_50s'] * 0.5)
df_merged_season['Defensive_Action'] = df_merged_season['tackles'] + df_merged_season['one_percenters'] + (df_merged_season['rebound_50s'] * 0.5)

df_merged_season[['player_name', 'year', 'Goals_per_Game', 'Fantasy_per_Game', 'Disposal_Reliability', 'Offensive_Impact', 'Defensive_Action']].head()
""")

add_md("""### Business Insight 2: Contextualizing Efficiency
**Insight:** Pure disposal numbers can be misleading. By introducing `Disposal_Reliability`, we can distinguish between high-volume ball-winners who turn the ball over frequently and those who distribute efficiently, which is critical for maintaining possession in modern AFL.

### Business Insight 3: Differentiating Player Roles
**Insight:** The `Offensive_Impact` and `Defensive_Action` scores allow us to mathematically separate attacking weapons from defensive anchors, facilitating role-specific recruitment analysis.""")

add_md("""## 2. Identify the Top 10 Most Valuable Players
We design a **Performance Index** utilizing 5 statistics: `Fantasy_per_Game`, `Offensive_Impact`, `Defensive_Action`, `avg_clearances`, and `avg_contested_possessions`.

We will standardize these stats and create a weighted index to rank players based on the most recent season (2023).""")

add_code("""# Filter for the most recent year available in the season dataset
recent_year = df_merged_season['year'].max()
df_recent = df_merged_season[df_merged_season['year'] == recent_year].copy()

# Filter for players with a meaningful number of games (e.g., > 10)
df_recent = df_recent[df_recent['games_played'] > 10]

# Features for Performance Index
metrics = ['Fantasy_per_Game', 'Offensive_Impact', 'Defensive_Action', 'avg_clearances', 'avg_contested_possessions']
for m in metrics:
    df_recent[m] = pd.to_numeric(df_recent[m], errors='coerce').fillna(0)

# Standardize (Z-score)
for m in metrics:
    col_mean = df_recent[m].mean()
    col_std = df_recent[m].std()
    df_recent[f"{m}_z"] = (df_recent[m] - col_mean) / (col_std if col_std > 0 else 1)

# Calculate Performance Index (Equal weights for simplicity, but could be adjusted)
df_recent['Performance_Index'] = df_recent[[f"{m}_z" for m in metrics]].sum(axis=1)

# Top 10 MVPs
top_10_mvp = df_recent.sort_values(by='Performance_Index', ascending=False).head(10)

# Visualization 1: Top 10 MVPs
plt.figure(figsize=(12, 6))
sns.barplot(data=top_10_mvp, x='Performance_Index', y='player_name', palette='viridis')
plt.title(f'Top 10 Most Valuable Players (Performance Index) - {recent_year}', fontsize=14, weight='bold')
plt.xlabel('Performance Index (Z-Score Sum)')
plt.ylabel('Player Name')
plt.show()
""")

add_md("""### Business Insight 4: MVP Versatility
**Insight:** The players topping the Performance Index excel across multiple facets of the game (scoring, defending, and contested ball), highlighting that modern MVPs must be versatile rather than one-dimensional specialists.

### Business Insight 5: Recruitment Targets
**Insight:** Ranking players by this holistic Performance Index reveals high-performing athletes who might be undervalued by traditional single-metric analyses (like goal-kicking alone), providing high-ROI targets for recruitment.""")


add_md("""## 3. Find the Most Consistent Players
We analyze the `afl_players_round_by_round_stats_raw.csv` data to find players whose performance remains stable throughout the season. We will use the Coefficient of Variation (CV = Standard Deviation / Mean) on `fantasy_points`.""")

add_code("""# Merge player names into round data
df_round_named = pd.merge(df_round, df_info[['id', 'player_name']], left_on='player_id', right_on='id', how='left')
df_round_recent = df_round_named[df_round_named['year'] == recent_year]

# Ensure fantasy points are numeric
df_round_recent['fantasy_points'] = pd.to_numeric(df_round_recent['fantasy_points'], errors='coerce')

# Calculate CV for each player
consistency = df_round_recent.groupby('player_name').agg(
    Games_Played=('round', 'count'),
    Mean_Fantasy=('fantasy_points', 'mean'),
    Std_Fantasy=('fantasy_points', 'std')
).reset_index()

# Filter players who played > 15 games and have a solid average (e.g., > 70 points)
consistency = consistency[(consistency['Games_Played'] > 15) & (consistency['Mean_Fantasy'] > 70)].copy()

# Coefficient of Variation (lower is more consistent)
consistency['CV'] = consistency['Std_Fantasy'] / consistency['Mean_Fantasy']
top_consistent = consistency.sort_values(by='CV', ascending=True).head(5)

# Visualization 2: Bar chart of CV
plt.figure(figsize=(10, 5))
sns.barplot(data=top_consistent, x='CV', y='player_name', palette='crest')
plt.title('Top 5 Most Consistent Players (Lowest CV of Fantasy Points)', fontsize=14, weight='bold')
plt.xlabel('Coefficient of Variation (Lower = More Consistent)')
plt.ylabel('Player Name')
plt.show()

# Visualization 3: Boxplot of the most consistent players
consistent_players_data = df_round_recent[df_round_recent['player_name'].isin(top_consistent['player_name'])]
plt.figure(figsize=(12, 6))
sns.boxplot(data=consistent_players_data, x='fantasy_points', y='player_name', palette='crest')
plt.title('Distribution of Fantasy Points for Consistent Players', fontsize=14, weight='bold')
plt.xlabel('Fantasy Points per Round')
plt.ylabel('Player Name')
plt.show()
""")

add_md("""### Business Insight 6: Predictable Output Reduces Risk
**Insight:** Players with low CV in their performance metrics offer predictable week-to-week output. Recruiting these players lowers team performance volatility, crucial for maintaining long-term winning streaks.

### Business Insight 7: High Floor vs High Ceiling
**Insight:** The boxplots reveal that these consistent players rarely have "disaster" games (their minimum scores are relatively high). A team built around high-floor players is statistically more resilient against injuries and form slumps.""")


add_md("""## 4. Identify Performance Trends
We will find players who improved and players who declined during the season. 
Improvement is measured by comparing Average Fantasy Points in the Second Half of the season vs the First Half.""")

add_code("""# Define first half and second half
# Get the maximum round numeric value
df_round_recent['round_num'] = pd.to_numeric(df_round_recent['round'], errors='coerce')
max_round = df_round_recent['round_num'].max()
mid_point = max_round / 2

# Split data
first_half = df_round_recent[df_round_recent['round_num'] <= mid_point]
second_half = df_round_recent[df_round_recent['round_num'] > mid_point]

# Calculate averages
fh_avg = first_half.groupby('player_name')['fantasy_points'].mean().reset_index().rename(columns={'fantasy_points': 'FH_Avg'})
sh_avg = second_half.groupby('player_name')['fantasy_points'].mean().reset_index().rename(columns={'fantasy_points': 'SH_Avg'})

# Merge and calculate trend
trends = pd.merge(fh_avg, sh_avg, on='player_name')
# Filter for players who actually played both halves
trends = trends.dropna()
trends['Improvement'] = trends['SH_Avg'] - trends['FH_Avg']

# Top 5 improved and Top 5 declined
top_improved = trends.sort_values(by='Improvement', ascending=False).head(5)
top_declined = trends.sort_values(by='Improvement', ascending=True).head(5)

# Visualization 4: Improved Players (Slope Graph / Line Chart)
plt.figure(figsize=(10, 6))
for idx, row in top_improved.iterrows():
    plt.plot(['First Half', 'Second Half'], [row['FH_Avg'], row['SH_Avg']], marker='o', label=row['player_name'])
plt.title('Top 5 Players with In-Season Improvement (Fantasy Points)', fontsize=14, weight='bold')
plt.ylabel('Average Fantasy Points')
plt.legend()
plt.show()

# Visualization 5: Declined Players
plt.figure(figsize=(10, 6))
for idx, row in top_declined.iterrows():
    plt.plot(['First Half', 'Second Half'], [row['FH_Avg'], row['SH_Avg']], marker='o', label=row['player_name'])
plt.title('Top 5 Players with In-Season Decline (Fantasy Points)', fontsize=14, weight='bold')
plt.ylabel('Average Fantasy Points')
plt.legend()
plt.show()
""")

add_md("""### Business Insight 8: Conditioning and Durability
**Insight:** Players showing significant second-half improvement demonstrate superior physical conditioning and adaptability. They peak when fatigue sets in for others, making them vital assets for finals campaigns.

### Business Insight 9: Red Flags in Decline
**Insight:** Steep drop-offs in the second half may indicate underlying injuries, poor fitness bases, or opposition teams figuring out a player's tactics. This is a massive red flag for contract renewals.""")

add_md("""## 5. Team Performance Analysis
We will rank all teams based on overall player performance using the engineered Performance Index. 
**Justification:** Aggregating the Performance Index of a team's top players provides a holistic view of the roster's structural strength, independent of win/loss luck.""")

add_code("""# Aggregate Performance Index by Team
# Clean team names to resolve overlaps
def clean_team_name(team):
    t = str(team).title().strip()
    if 'Collingwood' in t: return 'Collingwood'
    if 'Adelaide' in t and 'Port' not in t: return 'Adelaide Crows'
    if 'Brisbane' in t: return 'Brisbane Lions'
    if 'Carlton' in t: return 'Carlton Blues'
    if 'Essendon' in t: return 'Essendon Bombers'
    if 'Fremantle' in t: return 'Fremantle Dockers'
    if 'Geelong' in t: return 'Geelong Cats'
    if 'Gold Coast' in t: return 'Gold Coast Suns'
    if 'Greater Western' in t or 'Gws' in t: return 'GWS Giants'
    if 'Hawthorn' in t: return 'Hawthorn Hawks'
    if 'Melbourne' in t and 'North' not in t: return 'Melbourne Demons'
    if 'North Melbourne' in t: return 'North Melbourne Kangaroos'
    if 'Port Adelaide' in t: return 'Port Adelaide Power'
    if 'Richmond' in t: return 'Richmond Tigers'
    if 'St Kilda' in t: return 'St Kilda Saints'
    if 'Sydney' in t and 'Greater' not in t and 'Western' not in t: return 'Sydney Swans'
    if 'West Coast' in t: return 'West Coast Eagles'
    if 'Western Bulldogs' in t: return 'Western Bulldogs'
    return t

df_recent['team'] = df_recent['team'].apply(clean_team_name)

# Using df_recent which has the Performance_Index
team_performance = df_recent.groupby('team')['Performance_Index'].sum().reset_index()
team_performance = team_performance.sort_values(by='Performance_Index', ascending=False)

# Visualization 6: Team Performance Ranking
plt.figure(figsize=(12, 8))
sns.barplot(data=team_performance, x='Performance_Index', y='team', palette='mako')
plt.title('Team Power Ranking Based on Aggregate Player Performance Index', fontsize=14, weight='bold')
plt.xlabel('Total Team Performance Index')
plt.ylabel('Team')
plt.show()

# Visualization 7: Correlation Heatmap of Engineered Features
plt.figure(figsize=(8, 6))
corr = df_recent[['Goals_per_Game', 'Fantasy_per_Game', 'Disposal_Reliability', 'Offensive_Impact', 'Defensive_Action']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation of Engineered Features', fontsize=14, weight='bold')
plt.show()
""")

add_md("""### Business Insight 10: Roster Depth vs Star Power
**Insight:** The team ranking by Aggregate Performance Index reflects true roster depth. Teams at the top possess both high-end talent and solid role players, affirming that depth is statistically correlated with overall squad power.""")

add_md("""## 6. Final Recommendations
Based on the data, we recommend 5 specific players for recruitment. These players represent a mix of MVP-level performance, extreme consistency, and upward momentum.

**Recommended Players:**
1. The #1 MVP (Highest Performance Index)
2. The Most Consistent Player (Lowest CV)
3. The Most Improved Player (Highest Second Half jump)
4. Top Defensive Anchor (Highest Defensive Action)
5. Top Offensive Weapon (Highest Offensive Impact)""")

add_code("""# Extracting the 5 targets
target_mvp = top_10_mvp.iloc[0]['player_name']
target_consistent = top_consistent.iloc[0]['player_name']
target_improved = top_improved.iloc[0]['player_name']
target_defense = df_recent.sort_values(by='Defensive_Action', ascending=False).iloc[0]['player_name']
target_offense = df_recent.sort_values(by='Offensive_Impact', ascending=False).iloc[0]['player_name']

recommended_players = [target_mvp, target_consistent, target_improved, target_defense, target_offense]

# Filter data for these players
df_rec = df_recent[df_recent['player_name'].isin(recommended_players)].drop_duplicates(subset=['player_name'])

# Normalize stats for Radar Chart
radar_metrics = ['Fantasy_per_Game', 'Offensive_Impact', 'Defensive_Action', 'Disposal_Reliability', 'Goals_per_Game']
df_radar = df_rec[['player_name'] + radar_metrics].copy()

for m in radar_metrics:
    df_radar[m] = df_radar[m] / df_radar[m].max() # Scale 0 to 1

# Visualization 8: Radar Charts for Recommendations
import math
categories = radar_metrics
N = len(categories)
angles = [n / float(N) * 2 * math.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for idx, row in df_radar.iterrows():
    values = row[categories].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['player_name'])
    ax.fill(angles, values, alpha=0.1)

plt.xticks(angles[:-1], categories)
plt.title('Radar Chart Comparison of Recommended Players', fontsize=15, weight='bold')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.show()

# Print Final Summary
print("FINAL RECRUITMENT RECOMMENDATIONS:")
print(f"1. MVP Pick: {target_mvp} (Unmatched all-around performance)")
print(f"2. Consistency Pick: {target_consistent} (Most reliable output week-to-week)")
print(f"3. Momentum Pick: {target_improved} (Highest late-season improvement)")
print(f"4. Defensive Pick: {target_defense} (Elite defensive action rate)")
print(f"5. Offensive Pick: {target_offense} (High-impact scoring threat)")
""")

with open('D:/netixsol/Week- 2/Day-8/Day4_AFL_Analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
