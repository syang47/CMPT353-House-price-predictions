import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor


def read_data():
    listings_data = pd.read_csv('listings.csv.gz')
    amenities_data = pd.read_json('amenities-vancouver.json.gz', lines=True)
    return listings_data, amenities_data


def clean_amenities_data(amenities_data, amenities_required):

    #find unique amenities and the number of them to choose which are important for a traveller
    # print(amenities_data['amenity'].value_counts())

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

def clean_listings_data(listings_data, num_accomodates, num_bedrooms, max_price):
    #keep only the columns we need
    columns_needed = ['id', 'listing_url', 'name', 'description', 'picture_url', 'latitude', 'longitude', 'property_type', 'accommodates', 'bedrooms', 'beds', 'amenities', 'price']
    listings_data = listings_data[columns_needed]

    pd.set_option('mode.chained_assignment', None)

    # Change price from str to float
    listings_data['price'] = listings_data['price'].apply(lambda x: float(x.replace('$','').replace(',','')))

    #filter based on accomodates, bedrooms and price requirements
    listings_data = listings_data.loc[(listings_data['price'] <= max_price) & (listings_data['accommodates'] >= num_accomodates) & (listings_data['bedrooms'] >= num_bedrooms)]

    if listings_data.empty:
        print("\nNo listings found with current filters, Change the filters and Try Again!!\n")
        exit()

    listings_data.reset_index(drop = True , inplace = True)

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

def find_best_listing():
    #Read Data
    listings_data, amenities_data = read_data()

    # Change amenities here (updated the "restaurant" typo)
    amenities_required = ['restaurant', 'fast_food', 'cafe','bank','atm','pharmacy','bicycle_rental','fuel','pub','bar','car_sharing','car_rental','clinic','doctors','hospital','ice_cream','fountain','theatre','police','bus_station']

    # TODO:turn this into user input in the end 
    num_accomodates = 10
    num_bedrooms = 3
    max_price = 300 

    #Data Cleaning
    amenities_data_clean = clean_amenities_data(amenities_data, amenities_required)
    listings_data_clean = clean_listings_data(listings_data,num_accomodates,num_bedrooms,max_price)

    #add a column for number of amenities nearby to each listing
    listings_data_clean['num_amenities_nearby'] = listings_data_clean.apply(lambda x: num_amenities(x['latitude'], x['longitude'], amenities_data_clean), axis = 1)

    #add a column 'amenities_score' based on the number of different amenities nearby
    listings_data_clean['amenities_score'] = listings_data_clean['num_amenities_nearby'].apply(lambda x : ameneties_score(x))

    #sort based on amenities score
    listings_data_clean.sort_values(['amenities_score'], ascending = [False], inplace = True)
    print(listings_data_clean)

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

def run_ml():
    listings_data, amenities_data = read_data()
    listings_data_clean = clean_data_ML(listings_data)
    amenities_required = ['restaurant', 'fast_food', 'cafe','bank','atm','pharmacy','bicycle_rental','fuel','pub','bar','car_sharing','car_rental','clinic','doctors','hospital','ice_cream','fountain','theatre','police','bus_station']
    amenities_data_clean = clean_amenities_data(amenities_data, amenities_required)

    # print(listings_data_clean)

    X = listings_data_clean.drop('price',1)
    y = listings_data_clean['price']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    knn = KNeighborsRegressor(50)
    knn.fit(X_train, y_train)
    print("Knn Score:",knn.score(X_valid, y_valid))

    rf = RandomForestRegressor(100, max_depth=40)
    rf.fit(X_train, y_train)
    print("Random Forest Score:",rf.score(X_valid, y_valid))

    gb =  GradientBoostingRegressor()
    gb.fit(X_train, y_train)    
    print("Gradient Boosting Score:",gb.score(X_valid, y_valid))

    # Now we want to see if adding amenities score improves our model
    #add a column for number of amenities nearby to each listing
    listings_data_clean['num_amenities_nearby'] = listings_data_clean.apply(lambda x: num_amenities(x['latitude'], x['longitude'], amenities_data_clean), axis = 1)
    
    #add a column 'amenities_score' based on the number of different amenities nearby
    listings_data_clean['amenities_score'] = listings_data_clean['num_amenities_nearby'].apply(lambda x : ameneties_score(x))
    listings_data_clean = listings_data_clean.drop('num_amenities_nearby',1)

    X = listings_data_clean.drop('price',1)
    y = listings_data_clean['price']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    knn = KNeighborsRegressor(50)
    knn.fit(X_train, y_train)
    print("Knn Score with ameneties_score:",knn.score(X_valid, y_valid))

    rf = RandomForestRegressor(100, max_depth=40)
    rf.fit(X_train, y_train)
    print("Random Forest Score with amenities_score:",rf.score(X_valid, y_valid))

    gb =  GradientBoostingRegressor()
    gb.fit(X_train, y_train)    
    print("Gradient Boosting Score with amenities_score:",gb.score(X_valid, y_valid))


def main():
    #find_best_listing()
    run_ml()


if __name__ == "__main__":
    main()
