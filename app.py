import os
import pandas as pd
import joblib
from flask import Flask, request, render_template, jsonify
from models import load_model, load_preprocessor
from data_preprocessing import ensure_required_columns

app = Flask(__name__)

REQUIRED_COLUMNS = {
    'LBA', 'LBD', 'LBH', 'RecentFormHome', 'RecentFormAway', 'Referee', 'FTR_num',
    'PC>2.5', 'AvgAHA', 'AHCh', 'PAHH', 'B365CAHA', 'MaxAHA', 'PAHA', 'AvgCAHH',
    'B365CD', 'AvgCD', 'WHCA', 'MaxCAHH', 'MaxCAHA', 'WHCH', 'B365CH', 'PCAHH',
    'MaxD', 'AvgCA', 'VCCH', 'MaxCA', 'PCAHA', 'Avg>2.5', 'AvgA', 'AvgC>2.5',
    'B365C>2.5', 'PC<2.5', 'WHCD', 'VCCD', 'IWCA', 'B365C<2.5', 'VCCA', 'AvgAHH',
    'AvgD', 'MaxH', 'AvgC<2.5', 'AvgCH', 'Max<2.5', 'Max>2.5', 'MaxCD', 'BWCD',
    'IWCH', 'B365AHA', 'B365AHH', 'Time', 'P>2.5', 'Avg<2.5', 'AvgH', 'P<2.5',
    'B365CA', 'B365>2.5', 'BWCA', 'B365CAHH', 'IWCD', 'MaxC>2.5', 'AvgCAHA',
    'MaxCH', 'MaxA', 'BWCH', 'B365<2.5', 'MaxAHH', 'AHh', 'MaxC<2.5',
    'AF', 'AC', 'HST', 'AST', 'HF', 'HS', 'HC', 'AS',
    'BbAH', 'BbMxAHA', 'BbOU', 'BbAv<2.5', 'BbMx<2.5', 'BbAvAHA', 'Bb1X2', 'BbMxH',
    'BbAv>2.5', 'BbMxAHH', 'BbAvH', 'BbMxD', 'BbAvA', 'BbMx>2.5', 'BbMxA', 'BbAvD',
    'BbAvAHH', 'BbAHh'
}

years = ['2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']
leagues = ['E0', 'E1', 'E2', 'E3', 'EC', 'SC0', 'SP1', 'D2']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/')
def home():
    return "Hello, Render!"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    
@app.route('/fixtures')
def fixtures():
    return render_template('fixtures.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        start_year = request.form.get('start_year')
        end_year = request.form.get('end_year')
        league = request.form.get('league')
        home_team = request.form.get('home_team')
        away_team = request.form.get('away_team')
        model_choice = request.form.get('model')

        if not start_year or not end_year or not league or not home_team or not away_team or not model_choice:
            return render_template('prediction.html', error="Please provide all inputs.", years=years, leagues=leagues)

        all_data = []
        for year in years:
            if start_year <= year <= end_year:
                data_path = f"./data/{year}/all-euro-data-{year}.xlsx"
                if os.path.exists(data_path):
                    df = pd.read_excel(data_path, sheet_name=league)
                    all_data.append(df)

        if not all_data:
            return render_template('prediction.html', error="No data available for the selected years and league.", years=years, leagues=leagues)

        data = pd.concat(all_data)
        data = ensure_required_columns(data, REQUIRED_COLUMNS)

        # Extract historical match data between the two teams
        historical_matches = data[((data['HomeTeam'] == home_team) & (data['AwayTeam'] == away_team)) |
                                  ((data['HomeTeam'] == away_team) & (data['AwayTeam'] == home_team))]

        try:
            preprocessor = load_preprocessor()
            features = preprocessor.transform(data)
        except ValueError as e:
            return render_template('prediction.html', error=str(e), years=years, leagues=leagues)

        if features.ndim == 1:
            features = features.reshape(1, -1)

        model = load_model(model_choice)
        if model:
            prediction_prob = model.predict_proba(features)[0]
            prediction = {
                'home_win': prediction_prob[0],
                'draw': prediction_prob[1],
                'away_win': prediction_prob[2]
            }
            prediction_text = f"Home Win: {prediction['home_win']*100:.2f}%, Draw: {prediction['draw']*100:.2f}%, Away Win: {prediction['away_win']*100:.2f}%"
            return render_template('prediction.html', prediction=prediction, prediction_text=prediction_text,
                                   historical_matches=historical_matches.to_dict(orient='records'), years=years, leagues=leagues)
        else:
            return render_template('prediction.html', error="Selected model not found.", years=years, leagues=leagues)

    return render_template('prediction.html', years=years, leagues=leagues)

@app.route('/get_teams/<league>')
def get_teams(league):
    teams = set()
    for year in years:
        data_path = f"./data/{year}/all-euro-data-{year}.xlsx"
        if os.path.exists(data_path):
            df = pd.read_excel(data_path, sheet_name=league)
            teams.update(df['HomeTeam'].unique())
            teams.update(df['AwayTeam'].unique())

    return jsonify({'teams': sorted(teams)})

if __name__ == '__main__':
    app.run(debug=True)
