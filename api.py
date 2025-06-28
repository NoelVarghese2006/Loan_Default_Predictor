import predicting
from flask import Flask, jsonify, request

app = Flask(__name__)

# Sample data
books = [
    {"id": 1, "title": "Concept of Physics", "author": "H.C Verma"},
    {"id": 2, "title": "Gunahon ka Devta", "author": "Dharamvir Bharti"},
    {"id": 3, "title": "Problems in General Physsics", "author": "I.E Irodov"}
]

# Add a new book
@app.route('/api', methods=['POST'])
def add_book():
    x = request.json
    return predicting.predict(x['person_age'], x['person_income'], x['person_home_ownership'], x['person_emp_length'], x['loan_intent'], x['loan_grade'], x['loan_amnt'], x['loan_int_rate'], x['loan_percent_income'], x['cb_person_default_on_file'], x['cb_person_cred_hist_length']), 201

if __name__ == '__main__':
    app.run(debug=True)