import sqlite3
import json

conn = sqlite3.connect('optuna_study.db')
c = conn.cursor()

# Get study_id
c.execute("SELECT study_id FROM studies;")
study_id = c.fetchone()[0]

# Try to get direction (MAXIMIZE or MINIMIZE)
try:
    c.execute("SELECT direction FROM study_directions WHERE study_id=?;", (study_id,))
    direction = c.fetchone()[0]
except:
    direction = 'MAXIMIZE' # Default or guess

# Get best trial for RandomForest
c.execute("""
    SELECT t.trial_id, tv.value 
    FROM trials t
    JOIN trial_values tv ON t.trial_id = tv.trial_id
    JOIN trial_params tp ON t.trial_id = tp.trial_id
    WHERE t.study_id=? AND t.state='COMPLETE' AND tp.param_name='model_type'
""", (study_id,))
trials_with_model = c.fetchall()

# Since model_type might be categorical and stored as integer index, let's fetch the categorical mapping from the first trial that has model_type
c.execute("SELECT distribution_json FROM trial_params WHERE param_name='model_type' LIMIT 1;")
dist_str = c.fetchone()[0]
dist = json.loads(dist_str)
choices = dist.get('attributes', {}).get('choices', [])

# Find the integer value for 'RandomForest'
if 'RandomForest' in choices:
    rf_index = choices.index('RandomForest')
else:
    rf_index = -1

c.execute("""
    SELECT t.trial_id, tv.value 
    FROM trials t
    JOIN trial_values tv ON t.trial_id = tv.trial_id
    JOIN trial_params tp ON t.trial_id = tp.trial_id
    WHERE t.study_id=? AND t.state='COMPLETE' AND tp.param_name='model_type' AND tp.param_value=?
""", (study_id, rf_index))
rf_trials = c.fetchall()

if direction == 'MAXIMIZE':
    best_trial = max(rf_trials, key=lambda x: x[1])
else:
    best_trial = min(rf_trials, key=lambda x: x[1])

best_trial_id = best_trial[0]
best_value = best_trial[1]

# Get params
c.execute("SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=?;", (best_trial_id,))
params = c.fetchall()

param_dict = {}
for p in params:
    name = p[0]
    val = p[1]
    dist_str = p[2]
    
    try:
        dist = json.loads(dist_str)
        if dist.get('name') == 'CategoricalDistribution':
            choices = dist.get('attributes', {}).get('choices', [])
            val = choices[int(val)]
    except:
        pass
        
    param_dict[name] = val

print("Best Trial ID for RF:", best_trial_id)
print("Best Value for RF:", best_value)
print("Best Params for RF:", json.dumps(param_dict, indent=2))
