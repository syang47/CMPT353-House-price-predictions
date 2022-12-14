import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import kneighbors_graph
from sklearn.ensemble import GradientBoostingRegressor
import re
import sys


def read_data(in_directory):
    listings_data = pd.read_csv('listings.csv.gz')
    amenities_data = pd.read_json('amenities-vancouver.json.gz', lines=True)
    user_input1 = pd.read_csv(in_directory, sep=':\s', names=['Preference','Preference_Data'], engine='python')
    
    return listings_data, amenities_data, user_input1


def clean_amenities_data(amenities_data):

    #find unique amenities and the number of them to choose which are important for a traveller
    # print(amenities_data['amenity'].value_counts())

    # Change amenities here 
    amenities_required = ['restaurant', 'fast_food', 'cafe','bank','atm','pharmacy','bicycle_rental','fuel','pub','bar','car_sharing','car_rental','clinic','doctors','hospital','ice_cream','fountain','theatre','police','bus_station']
    #adapted from : https://www.kite.com/python/answers/how-to-filter-a-pandas-dataframe-with-a-list-by-%60in%60-or-%60not-in%60-in-python
    bool_series = amenities_data.amenity.isin(amenities_required)
    filtered_amenities_df = amenities_data[bool_series]
    filtered_amenities_df = filtered_amenities_df.drop(['timestamp','tags'], axis=1).dropna() # dropping unnecessary columns, and filter out NA values
    filtered_amenities_df.reset_index(inplace=True, drop=True)

    return filtered_amenities_df


#reference: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
def haversine_distance(df, lon2, lat2):
    # convert decimal degrees to radians 
    lon1=np.radians(df['lon'])
    lat1=np.radians(df['lat'])
    lon2=np.radians(lon2)
    lat2=np.radians(lat2)
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = (dlat/2).apply(sin)**2 + (lat1).apply(sin) * cos(lat2) * (dlon/2).apply(sin)**2
    c = 2 * ((a).apply(sqrt).apply(asin)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000


# handles returns converted input values from file 
def handle_input(inputfile):
    for (x) in range(len(inputfile)):
        if inputfile['Preference'].iloc[x].lower() == "accommodates":
            accommodates_input = float(inputfile['Preference_Data'].iloc[x])
        if inputfile['Preference'].iloc[x].lower() == "bedrooms":
            room_input = float(inputfile['Preference_Data'].iloc[x])
        if inputfile['Preference'].iloc[x].lower() == "price range":
            price_range_input = inputfile['Preference_Data'].iloc[x]
        if inputfile['Preference'].iloc[x].lower() == "exact":
            exact = inputfile['Preference_Data'].iloc[x]
    if exact.lower() == "true":
        exact = True
    else:
        exact = False
           
    return accommodates_input, room_input, price_range_input, exact

# returns cleaned data listings
def clean_listings_data(listings_data, accommodates_input, room_input, price_range_input, exact):
    #keep only the columns we need
    columns_needed = ['id', 'listing_url', 'name', 'description', 'picture_url', 'latitude', 'longitude', 'property_type', 'accommodates', 'bedrooms', 'beds', 'amenities', 'price']
    listings_data = listings_data[columns_needed].copy()
    listings_data['price'] = listings_data['price'].apply(lambda x: x.replace(',','').replace('$','')).astype(float)
    
    pd.set_option('mode.chained_assignment', None)
    # extract price_range from string
    p_range = [float(s) for s in re.findall('[0-9]+', price_range_input)]
    min_price = p_range[0]
    max_price = p_range[1]
    
    bedrooms=listings_data['bedrooms']
    accommodates=listings_data['accommodates']
    
    # if user is fine with referencing their input as minimum requirements -> if not exact
    # if user wants exact filter -> else 
    
    if not exact:
        # find listing data with bedrooms >= room_input and accommodates >= accomodates_input
        listings_data = listings_data[(bedrooms >= room_input) & (accommodates >= accommodates_input)]
        listings_data = listings_data[(listings_data['price'] <= max_price) & (listings_data['price'] >= min_price)]
    else:
        # find listing data with bedrooms == room_input and accommodates == accomodates_input
        listings_data = listings_data[(bedrooms == room_input) & (accommodates == accommodates_input)].reset_index(drop=True)
        listings_data = listings_data[(listings_data['price'] <= max_price) & (listings_data['price'] >= min_price)]
    

    listings_data=listings_data.reset_index(drop=True)
    
    return listings_data


# return a dictionary with number of amenities in a 1km radius of this lat and lon
def num_amenities(lat, lon, amenities_data_clean):
    distance = haversine_distance(amenities_data_clean, lon, lat)
    amenities_data_clean['distance'] = distance
    data_withinR = amenities_data_clean.loc[amenities_data_clean['distance'] < 1000].reset_index(drop=True)
    amenities_series = data_withinR.pivot_table(columns = ['amenity'], aggfunc='size')  # Counts # of amenities, type=pd.series
    amenities_dict = amenities_series.to_dict()# converts series to dict
    return amenities_dict
    

def ameneties_score(my_dict):
    
    num_different_amenities = len(my_dict)
    score = num_different_amenities * 10 
        
    for key in my_dict:
        if (my_dict[key] > 30):
            score+=30
        else:
            score+= my_dict[key]
    return score

def find_best_listing(listings_data_clean, amenities_data_clean):
    
    # print(listings_data_clean)

    # print(amenities_data_clean)

    #add a column for number of amenities nearby to each listing
    listings_data_clean['num_amenities_nearby'] = listings_data_clean.apply(lambda x: num_amenities(x['latitude'], x['longitude'], amenities_data_clean), axis = 1)

    #add a column 'amenities_score' based on the number of different amenities nearby
    listings_data_clean['amenities_score'] = listings_data_clean['num_amenities_nearby'].apply(lambda x : ameneties_score(x))

    #sort based on amenities score
    listings_data_clean.sort_values(['amenities_score'], ascending = [False], inplace = True)
    listings_data_clean.reset_index(drop = True, inplace = True)

    return listings_data_clean


def clean_data_ML(listings_data):
    
    columns_needed = ['latitude', 'longitude', 'host_response_time', 'host_response_rate', 'host_acceptance_rate','host_is_superhost','host_listings_count', 'host_total_listings_count', 'host_identity_verified','neighbourhood_cleansed', 'property_type', 'room_type', 'accommodates', 'bedrooms', 'beds', 'amenities', 'price',   'minimum_nights', 'maximum_nights', 'maximum_nights_avg_ntm',  'availability_30', 'availability_60', 'availability_90','availability_365','number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value', 'reviews_per_month' ]
    listings_data = listings_data[columns_needed]

    # remove all rows with any Null Value
    listings_data = listings_data.dropna(how='any',axis=0) 

    # find number of amenities provided by the host
    listings_data['num_amenities'] = listings_data['amenities'].apply(lambda x: len(x)).astype('float64')
    listings_data.drop('amenities', axis= 1, inplace = True)

    #Label Encoding categorical Data
    lb_make = LabelEncoder()
    listings_data['host_response_time'] = lb_make.fit_transform(listings_data['host_response_time'])
    listings_data['host_is_superhost'] = lb_make.fit_transform(listings_data['host_is_superhost'])
    listings_data['host_identity_verified'] = lb_make.fit_transform(listings_data['host_identity_verified'])
    listings_data['neighbourhood_cleansed'] = lb_make.fit_transform(listings_data['neighbourhood_cleansed'])
    listings_data['property_type'] = lb_make.fit_transform(listings_data['property_type'])
    listings_data['room_type'] = lb_make.fit_transform(listings_data['room_type'])

    #convert strings to float
    listings_data['host_response_rate'] = listings_data['host_response_rate'].apply(lambda x: float(x.replace('%','')))
    listings_data['host_acceptance_rate'] = listings_data['host_acceptance_rate'].apply(lambda x: float(x.replace('%','')))
    listings_data['price'] = listings_data['price'].apply(lambda x: float(x.replace('$','').replace(',','')))
    return listings_data

def run_ml(listings_data_clean, amenities_data_clean):

    #add a column for number of amenities nearby to each listing
    listings_data_clean['num_amenities_nearby'] = listings_data_clean.apply(lambda x: num_amenities(x['latitude'], x['longitude'], amenities_data_clean), axis = 1)
    
    #add a column 'amenities_score' based on the number of different amenities nearby
    listings_data_clean['amenities_score'] = listings_data_clean['num_amenities_nearby'].apply(lambda x : ameneties_score(x))
    listings_data_clean = listings_data_clean.drop(['num_amenities_nearby'],1)
    
    X = listings_data_clean.drop('price',1)
    y = listings_data_clean['price']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    X_train_without_amenity_score = X_train.drop('amenities_score', 1)
    X_valid_without_amenity_score = X_valid.drop('amenities_score', 1)

    knn = KNeighborsRegressor(50)
    knn.fit(X_train_without_amenity_score, y_train)
    knn_sc = knn.score(X_valid_without_amenity_score, y_valid)
    print("Knn Score:", knn_sc)
    
    rf = RandomForestRegressor(100, max_depth=40)
    rf.fit(X_train_without_amenity_score, y_train)
    rf_sc = rf.score(X_valid_without_amenity_score, y_valid)
    print("Random Forest Score:",rf_sc)
    
    gb =  GradientBoostingRegressor()
    gb.fit(X_train_without_amenity_score, y_train)    
    gb_sc = gb.score(X_valid_without_amenity_score, y_valid)
    print("Gradient Boosting Score:",gb_sc)

    # Now we want to see if adding amenities score improves our model
    knn_A = KNeighborsRegressor(50)
    knn_A.fit(X_train, y_train)
    knn_A_sc = knn_A.score(X_valid, y_valid)
    print("\nKnn Score with ameneties_score:",knn_A_sc)

    rf_A = RandomForestRegressor(100, max_depth=40)
    rf_A.fit(X_train, y_train)
    rf_A_sc = rf_A.score(X_valid, y_valid)
    print("Random Forest Score with amenities_score:",rf_A_sc)

    gb_A = GradientBoostingRegressor()
    gb_A.fit(X_train, y_train)  
    gb_A_sc = gb_A.score(X_valid, y_valid)
    print("Gradient Boosting Score with amenities_score:",gb_A_sc)

    return knn_sc, knn_A_sc, rf_sc, rf_A_sc, gb_sc, gb_A_sc


def main(in_directory):

    # Find the best listings based on User Requirement and Amenities nearby Airbnb
    listings_data, amenities_data, user_input1 = read_data(in_directory)

    #clean amenities data
    amenities_data_clean = clean_amenities_data(amenities_data)

    #clean listings data data
    accommodates_input, room_input, price_range_input, exact = handle_input(user_input1)
    listings_data_clean = clean_listings_data(listings_data, accommodates_input, room_input, price_range_input, exact)

    if listings_data_clean.empty:
        print("No listings found for these filters, Change fiilters and Try Again")
    else:
        best_listings = find_best_listing(listings_data_clean, amenities_data_clean)

        # print(best_listings)

        best_listings.to_csv("AirBnb_search_results.csv", na_rep='(missing)')

    print("\n Running ML models...\n")
    #ML Part
    listings_data_clean_ml = clean_data_ML(listings_data)
    [knn_sc, knn_A_sc, rf_sc, rf_A_sc, gb_sc, gb_A_sc] = run_ml(listings_data_clean_ml, amenities_data_clean)
    ML_RES = [[knn_sc,knn_A_sc], 
            [rf_sc, rf_A_sc], 
            [gb_sc,gb_A_sc]]
    ML_df = pd.DataFrame(ML_RES, 
                         columns=["AirBnb's Listing Info", "AirBnb's Listing Info with Amenity Scores"], 
                         index=['K-Nearest Neighbors','Random Forest','Gradient Boosting'])
    ML_df.index.name = "Regressors Used"

    # outputting the ML prediction results to csv file
    ML_OUT = ML_df.to_csv("ML_Price_Prediction.csv",na_rep='(missing')


if __name__ == "__main__":
    in_directory = sys.argv[1]
    main(in_directory)
