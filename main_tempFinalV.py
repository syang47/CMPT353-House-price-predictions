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

def read_data():
    listings_data = pd.read_csv('listings.csv.gz')
    amenities_data = pd.read_json('amenities-vancouver.json.gz', lines=True)
    user_input1 = pd.read_csv('input1.txt', sep=':\s', names=['Preference','Preference_Data'], engine='python')
    
    return listings_data, amenities_data, user_input1


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

# 

def clean_listings_data(listings_data, accommodates_input, room_input, price_range_input, exact, amenities_data_clean):
    #keep only the columns we need
    columns_needed = ['id', 'listing_url', 'name', 'description', 'picture_url', 'latitude', 'longitude', 'property_type', 'accommodates', 'bedrooms', 'beds', 'amenities', 'price',  'review_scores_value']
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
    
    # if listings_data is empty
    if listings_data.empty:
        print("Cannot find any listings with current filter, please try with other filters.\n")
        return   
    #add a column for number of amenities nearby to each listing
    listings_data['num_amenities_nearby'] = listings_data.apply(lambda x: num_amenities(x['latitude'], x['longitude'], amenities_data_clean), axis = 1)

    #add a column 'amenities_score' based on the number of different amenities nearby
    listings_data['amenities_score'] = listings_data['num_amenities_nearby'].apply(lambda x : ameneties_score(x))
    
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

    
def run_ml(listings_data_clean, amenities_data_clean):
    
    X = listings_data_clean.drop('price',1)
    y = listings_data_clean['price']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    knn = KNeighborsRegressor(n_neighbors=50)
    knn.fit(X_train, y_train)
    knn_sc = knn.score(X_valid, y_valid)
    print("Knn Score:", knn_sc)
    
    rf = RandomForestRegressor(100, max_depth=40)
    rf.fit(X_train, y_train)
    rf_sc = rf.score(X_valid, y_valid)
    print("Random Forest Score:",rf_sc)
    
    gb =  GradientBoostingRegressor()
    gb.fit(X_train, y_train)    
    gb_sc = gb.score(X_valid, y_valid)
    print("Gradient Boosting Score:",gb_sc)

    # Now we want to see if adding amenities score improves our model
    #add a column for number of amenities nearby to each listing
    listings_data_clean['num_amenities_nearby'] = listings_data_clean.apply(lambda x: num_amenities(x['latitude'], x['longitude'], amenities_data_clean), axis = 1)
    
    #add a column 'amenities_score' based on the number of different amenities nearby
    listings_data_clean['amenities_score'] = listings_data_clean['num_amenities_nearby'].apply(lambda x : ameneties_score(x))
    listings_data_clean = listings_data_clean.drop('num_amenities_nearby',1)

    X = listings_data_clean.drop('price',1)
    y = listings_data_clean['price']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    knn_A = KNeighborsRegressor(50)
    knn_A.fit(X_train, y_train)
    knn_A_sc = knn_A.score(X_valid, y_valid)
    print("Knn Score with ameneties_score:",knn_A_sc)

    rf_A = RandomForestRegressor(100, max_depth=40)
    rf_A.fit(X_train, y_train)
    rf_A_sc = rf_A.score(X_valid, y_valid)
    print("Random Forest Score with amenities_score:",rf_A_sc)

    gb_A = GradientBoostingRegressor()
    gb_A.fit(X_train, y_train)  
    gb_A_sc = gb_A.score(X_valid, y_valid)
    print("Gradient Boosting Score with amenities_score:",gb_A_sc)

    return knn_sc, knn_A_sc, rf_sc, rf_A_sc, gb_sc, gb_A_sc

# handles user's input textfile 
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

# generates top 3 listings sorted by best review score, best amenity score, and best price
def generate_output(listings_data_clean):
    
    # sort by price in ascending order
    listings_by_price = listings_data_clean.sort_values(by="price", ascending=True).reset_index(drop=True)

    # sort by review_scores_values in descending order, drop na scores(?)
    listings_by_rscore = listings_data_clean.sort_values(['review_scores_value'], ascending=False).dropna().reset_index(drop=True)

    # sort by amenities_score in descending order, drop na scores(?)
    listings_by_ascore = listings_data_clean.sort_values(['amenities_score'], ascending=False).dropna().reset_index(drop=True)
    
    # output lowest_price listing, highest scored listing to another dataframe
    x = listings_by_rscore.head(1)
    x["result"] = "Best Scored"    
    cols = list(x.columns)
    cols = [cols[-1]]+cols[:-1]
    x = x[cols]

    y = listings_by_price.head(1)
    y["result"] = "Best Valued"
    cols = list(y.columns)
    cols = [cols[-1]]+cols[:-1]
    y = y[cols]

    z = listings_by_ascore.head(1)
    z["result"] = "Best Amenity Scored"
    cols = list(z.columns)
    cols = [cols[-1]]+cols[:-1]
    z = z[cols]

    sorted_output = pd.concat([z, x, y], axis=0, ignore_index=True)  
    return sorted_output

# returns amenities around the top 3 listings
def find_amenties_around_top_listings(sorted_output, amenities_data_clean, i):
    amen_output = amenities_data_clean.copy()
    lon = sorted_output.iloc[i,7] # lon from utput listing 
    lat = sorted_output.iloc[i,6] # lat
    distance = haversine_distance(amen_output, lon, lat)
    amen_output['distance'] = distance
    data_withinR = amen_output.loc[amen_output['distance'] < 1000].reset_index(drop=True)
        
    return data_withinR

def main():
    # read data
    listings_data, amenities_data, user_input1 = read_data()
    
    # handle input file
    
    accommodates_input, room_input, price_range_input, exact = handle_input(user_input1)
    

    # clean OSM amenities data
    amenities_required = ['restaurant', 'fast_food', 'cafe','bank','atm','pharmacy','bicycle_rental','fuel','pub','bar','car_sharing','car_rental','clinic','doctors','hospital','ice_cream','fountain','theatre','police','bus_station']
    amenities_data_clean = clean_amenities_data(amenities_data, amenities_required)
    print(amenities_data_clean)
    
    # clean AirBnb listings data
    listings_data_clean = clean_listings_data(listings_data, accommodates_input, room_input, price_range_input, exact, amenities_data_clean)

    # Perform ML Trials and store output score to df
    [knn_sc, knn_A_sc, rf_sc, rf_A_sc, gb_sc, gb_A_sc] = run_ml(clean_data_ML(listings_data), amenities_data_clean)
    ML_RES = [[knn_sc,knn_A_sc], 
              [rf_sc, rf_A_sc], 
              [gb_sc,gb_A_sc]]
    ML_df = pd.DataFrame(ML_RES, 
                         columns=["AirBnb's Listing Info", "AirBnb's Listing Info with Amenity Scores"], 
                         index=['K-Nearest Neighbors','Random Forest','Gradient Boosting'])
    ML_df.index.name = "Regressors Used"
    print(ML_df)

    # sort filtered data by best review score, best amenity score, and best price
    sorted_output = generate_output(listings_data_clean)

    # outputting the filtered listings --> top 3 and total
    TOP3_OUT = sorted_output.to_csv("Top3_Filtered_ABNB_Listings.csv",na_rep='(missing)')
    TOTAL_OUT = listings_data_clean.to_csv("Total_Filtered_ABNB_Listings.csv",na_rep='(missing)')
    
    # outputting nearby amenities around the top 3 filtered listing
    for i in range(len(sorted_output)):
        if sorted_output['result'].iloc[i] == "Best Amenity Scored":
            data_withinA = find_amenties_around_top_listings(sorted_output, amenities_data_clean, i)
            data_withinA.index.name = "Best Amenity Scored"
            data_withinA.to_csv('Best_Amenity_Scored_Listing_nearbyAmenities.csv',na_rep='(missing)')
            print("Found the Best Amenity Scored Listing's nearby amenities")

        if sorted_output['result'].iloc[i] == "Best Scored":
            data_withinS = find_amenties_around_top_listings(sorted_output, amenities_data_clean, i)
            data_withinS.index.name = "Best Scored"
            data_withinS.to_csv('Best_Reviewd_Scored_Listing_nearbyAmenities.csv',na_rep='(missing)')
            print("Found the Best Review Scored Listing's nearby amenities")

        if sorted_output['result'].iloc[i] == "Best Valued":
            data_withinP = find_amenties_around_top_listings(sorted_output, amenities_data_clean, i)
            data_withinA.index.name = "Best Valued"
            data_withinA.to_csv('Best_Valued_Listing_nearbyAmenities.csv',na_rep='(missing)')
            print("Found the Best Valued Listing's nearby amenities")

    # outputting the ML prediction results
    ML_OUT = ML_df.to_csv("ML_Price_Prediction.csv",na_rep='(missing)')

    
if __name__ == "__main__":
    main()
