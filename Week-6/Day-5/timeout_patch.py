import re
with open('d:/netixsol/Week-6/Day-5/afl_assistant_core.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Retrieval node
code = code.replace('result = get_team_h2h_record(ta, tb)', 'result = run_with_timeout(get_team_h2h_record, args=(ta, tb), timeout_sec=10.0)')
code = code.replace('result = get_player_season_stats(int(pid), int(year))', 'result = run_with_timeout(get_player_season_stats, args=(int(pid), int(year)), timeout_sec=10.0)')
code = code.replace('result = retrieve_afl_knowledge(query)', 'result = run_with_timeout(retrieve_afl_knowledge, args=(query,), timeout_sec=10.0)')

# DirectAnswer node
code = code.replace('answer = chat_with_agent(query, history_tuples)', 'answer = run_with_timeout(chat_with_agent, args=(query, history_tuples), timeout_sec=15.0)')

# Prediction node match winner
code = code.replace('''return predict_match_winner_tool.invoke({
            "home_team":        team_a or "Richmond Tigers",
            "away_team":        team_b or "Collingwood Magpies",
            "temporal_context": temporal,
        })''', '''return run_with_timeout(
            predict_match_winner_tool.invoke,
            args=({
                "home_team":        team_a or "Richmond Tigers",
                "away_team":        team_b or "Collingwood Magpies",
                "temporal_context": temporal,
            },),
            timeout_sec=10.0
        )''')

# Prediction node top player
code = code.replace('''return predict_top_player_tool.invoke({
            "team":             team,
            "temporal_context": temporal,
            "stat_type":        stat or "cpi",
            "top_k":            5,
        })''', '''return run_with_timeout(
            predict_top_player_tool.invoke,
            args=({
                "team":             team,
                "temporal_context": temporal,
                "stat_type":        stat or "cpi",
                "top_k":            5,
            },),
            timeout_sec=10.0
        )''')

with open('d:/netixsol/Week-6/Day-5/afl_assistant_core.py', 'w', encoding='utf-8') as f:
    f.write(code)
print('Applied timeouts')
