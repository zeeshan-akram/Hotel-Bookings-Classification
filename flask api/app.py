from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pickle

app = Flask(__name__)




def get_categories(data):
    c_f = {
        'hotel' : ['Resort Hotel'],
        'meal' : ['FB', 'HB', 'SC', 'Undefined'],
        'country' : ['AGO', 'AIA', 'ALB', 'AND', 'ARE', 'ARG', 'ARM', 'ASM', 'ATA', 'ATF', 'AUS', 'AUT', 'AZE', 
                    'BDI', 'BEL', 'BEN', 'BFA', 'BGD', 'BGR', 'BHR', 'BHS', 'BIH', 'BLR', 'BOL', 'BRA', 'BRB', 
                    'BWA', 'CAF', 'CHE', 'CHL', 'CHN', 'CIV', 'CMR', '_CN', 'COL', 'COM', 'CPV', 'CRI', 'CUB', 
                    'CYM', 'CYP', 'CZE', 'DEU', 'DJI', 'DMA', 'DNK', 'DOM', 'DZA', 'ECU', 'EGY', 'ESP', 'EST', 
                    'ETH', 'FIN', 'FJI', 'FRA', 'FRO', 'GAB', 'GBR', 'GEO', 'GGY', 'GHA', 'GIB', 'GLP', 'GNB', 
                    'GRC', 'GTM', 'GUY', 'HKG', 'HND', 'HRV', 'HUN', 'IDN', 'IMN', 'IND', 'IRL', 'IRN', 'IRQ',
                    'ISL', 'ISR', 'ITA', 'JAM', 'JEY', 'JOR', 'JPN', 'KAZ', 'KEN', 'KHM', 'KIR', 'KNA', 'KOR', 
                    'KWT', 'LAO', 'LBN', 'LBY', 'LCA', 'LIE', 'LKA', 'LTU', 'LUX', 'LVA', 'MAC', 'MAR', 'MCO', 
                    'MDG', 'MDV', 'MEX', 'MKD', 'MLI', 'MLT', 'MMR', 'MNE', 'MOZ', 'MRT', 'MUS', 'MWI', 'MYS',
                    'MYT', 'NAM', 'NCL', 'NGA', 'NIC', 'NLD', 'NOR', 'NPL', 'NZL', 'OMN', 'PAK', 'PAN', 'PER',
                    'PHL', 'PLW', 'POL', 'PRI', 'PRT', 'PRY', 'PYF', 'QAT', 'ROU', 'RUS', 'RWA', 'SAU', 'SDN', 
                    'SEN', 'SGP', 'SLE', 'SLV', 'SMR', 'SRB', 'STP', 'SUR', 'SVK', 'SVN', 'SWE', 'SYC', 'SYR',
                    'TGO', 'THA', 'TJK', 'TMP', 'TUN', 'TUR', 'TWN', 'TZA', 'UGA', 'UKR', 'UMI', 'URY', 'USA',
                    'UZB', 'VEN', 'VGB', 'VNM', 'ZAF', 'ZMB', 'ZWE', 'unknown'],
        'market_segment' : ['Complementary', 'Corporate', 'Direct', 'Groups', 'Offline TA/TO', 'Online TA', 'Undefined'],
        'distribution_channel' : ['Direct', 'GDS', 'TA/TO', 'Undefined'],
        'reserved_room_type' : ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'L', 'P'],
        'assigned_room_type' : ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'P'],
        'deposit_type' : ['Non Refund', 'Refundable'],
        'customer_type' : ['Group', 'Transient', 'Transient-Party']
    }

    cate_df = pd.DataFrame()
    for col in c_f.keys():
        category = data[col].values
        if category in c_f[col]:
            for cat in c_f[col]:
                if cat == category:
                    cate_df[col + f'_{cat}'] = [1]
                else:
                    cate_df[col + f'_{cat}'] = [0]
        else:
            for cat in c_f[col]:
                cate_df[col + f'_{cat}'] = [0]
    return cate_df


def transform_data(data):
    num_fea = ['lead_time', 'previous_cancellations', 
             'booking_changes', 'required_car_parking_spaces', 'total_of_special_requests']
    num_data = data[num_fea]
    for col in num_data.columns:
        num_data[col] = num_data[col].astype('int64')
    min_scale = pickle.load(open('variable_scalers.pkl', 'rb'))
    num_data = pd.DataFrame(min_scale.transform(num_data), columns=num_fea)
    cate_data = get_categories(data)
    final_data = pd.concat([num_data, cate_data], axis=1)
    return final_data


def make_predictions(data):

    final_model = pickle.load(open("booking_hotel_model.pkl", "rb"))
    prediction = final_model.predict(data)
    return round(prediction[0])




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/countries')
def countries():
    return render_template('countries.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    cntry_list = ['PRT', 'GBR', 'USA', 'ESP', 'IRL', 'FRA', 'unknown', 'ROU', 'NOR',
       'OMN', 'ARG', 'POL', 'DEU', 'BEL', 'CHE', 'CN', 'GRC', 'ITA',
       'NLD', 'DNK', 'RUS', 'SWE', 'AUS', 'EST', 'CZE', 'BRA', 'FIN',
       'MOZ', 'BWA', 'LUX', 'SVN', 'ALB', 'IND', 'CHN', 'MEX', 'MAR',
       'UKR', 'SMR', 'LVA', 'PRI', 'SRB', 'CHL', 'AUT', 'BLR', 'LTU',
       'TUR', 'ZAF', 'AGO', 'ISR', 'CYM', 'ZMB', 'CPV', 'ZWE', 'DZA',
       'KOR', 'CRI', 'HUN', 'ARE', 'TUN', 'JAM', 'HRV', 'HKG', 'IRN',
       'GEO', 'AND', 'GIB', 'URY', 'JEY', 'CAF', 'CYP', 'COL', 'GGY',
       'KWT', 'NGA', 'MDV', 'VEN', 'SVK', 'FJI', 'KAZ', 'PAK', 'IDN',
       'LBN', 'PHL', 'SEN', 'SYC', 'AZE', 'BHR', 'NZL', 'THA', 'DOM',
       'MKD', 'MYS', 'ARM', 'JPN', 'LKA', 'CUB', 'CMR', 'BIH', 'MUS',
       'COM', 'SUR', 'UGA', 'BGR', 'CIV', 'JOR', 'SYR', 'SGP', 'BDI',
       'SAU', 'VNM', 'PLW', 'QAT', 'EGY', 'PER', 'MLT', 'MWI', 'ECU',
       'MDG', 'ISL', 'UZB', 'NPL', 'BHS', 'MAC', 'TGO', 'TWN', 'DJI',
       'STP', 'KNA', 'ETH', 'IRQ', 'HND', 'RWA', 'KHM', 'MCO', 'BGD',
       'IMN', 'TJK', 'NIC', 'BEN', 'VGB', 'TZA', 'GAB', 'GHA', 'TMP',
       'GLP', 'KEN', 'LIE', 'GNB', 'MNE', 'UMI', 'MYT', 'FRO', 'MMR',
       'PAN', 'BFA', 'LBY', 'MLI', 'NAM', 'BOL', 'PRY', 'BRB', 'ABW',
       'AIA', 'SLV', 'DMA', 'PYF', 'GUY', 'LCA', 'ATA', 'GTM', 'ASM',
       'MRT', 'NCL', 'KIR', 'SDN', 'ATF', 'SLE', 'LAO']
    
    if request.form['country'] not in cntry_list:
        return render_template('no_country.html')
    

    data = pd.DataFrame()
    features = ['lead_time', 'previous_cancellations', 'booking_changes',
       'required_car_parking_spaces', 'total_of_special_requests', 'hotel',
       'meal', 'country', 'market_segment', 'distribution_channel',
       'reserved_room_type', 'assigned_room_type', 'deposit_type',
       'customer_type']

    for index in range(len(features)):
           value = request.form[features[index]]
           data[features[index]] = [value]
    
    final_data = transform_data(data)

    return render_template('predictions.html', predictions= make_predictions(final_data))

if __name__ == "__main__":
    app.run(debug=True)