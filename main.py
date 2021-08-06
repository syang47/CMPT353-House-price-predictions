import pandas as pd
import numpy as np
import re
from math import radians, cos, sin, asin, sqrt

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

def clean_listings_data(listings_data, accommodates_input, room_input, price_range_input, exact):
    #keep only the columns we need
    columns_needed = ['id', 'listing_url', 'name', 'description', 'picture_url', 'latitude', 'longitude', 'property_type', 'accommodates', 'bedrooms', 'beds', 'amenities', 'price',  'review_scores_value']
    listings_data = listings_data[columns_needed].copy()
    listings_data['price'] = listings_data['price'].apply(lambda x: x.replace(',','').replace('$','')).astype(float)
    
    pd.set_option('mode.chained_assignment', None)
    # extract price_range from string
    p_range = [float(s) for s in re.findall('[^-]?\d[\d.,]+', price_range_input)]
    min_price = p_range[0]
    max_price = p_range[1]
    
    bedrooms=listings_data['bedrooms']
    accommodates=listings_data['accommodates']
    if not exact:
        # find listing data with bedrooms >= room_input and accommodates >= accomodates_input
        listings_data = listings_data[(bedrooms >= room_input) & (accommodates >= accommodates_input)].reset_index(drop=True)
        listings_data = listings_data[(listings_data['price'] <= max_price) & (listings_data['price'] >= min_price)].reset_index(drop=True)
    
    else:
        # find listing data with bedrooms == room_input and accommodates == accomodates_input
        listings_data = listings_data[(bedrooms == room_input) & (accommodates == accommodates_input)].reset_index(drop=True)
        listings_data = listings_data[(listings_data['price'] <= max_price) & (listings_data['price'] >= min_price)].reset_index(drop=True)
    if listings_data.empty:
        print("Cannot find any listings with current filter, please try with other filters.\n")
        return
    
    
    
    return listings_data

# haven't fixed bugs yet
def output_by_sort(listing_data):
    # sort by price in ascending order
    listings_by_price = listings_data.sort_values(by="price", ascending=True).reset_index(drop=True)

    # sort by review_scores_values in descending order, drop na scores(?)
    listings_by_rscore = listings_data.sort_values(['review_scores_value'], ascending=False).dropna().reset_index(drop=True)

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
    
    temp_output = pd.concat([x, y], axis=0, ignore_index=True)
    return temp_output

# return a dictionary with number of amenities in a 1km radius of this lat and lon
def num_amenities(lat, lon, amenities_data_clean):
    distance = haversine_distance(amenities_data_clean, lon, lat)
    amenities_data_clean['distance'] = distance
    data_withinR = amenities_data_clean.loc[amenities_data_clean['distance'] < 1000].reset_index(drop=True)
    amenities_series = data_withinR.pivot_table(columns = ['amenity'], aggfunc='size')  # Counts # of amenities, type=pd.series
    amenities_dict = amenities_series.to_dict()# converts series to dict
    return amenities_dict

def main():

    #Read Data
    listings_data, amenities_data = read_data()

    # Change amenities here (updated the "restaurant" typo)
    amenities_required = ['restaurant', 'fast_food', 'cafe','bank','atm','pharmacy','bicycle_rental','fuel','pub','bar','car_sharing','car_rental','clinic','doctors','hospital','ice_cream','fountain','theatre','police','bus_station']

    # User Input
    ## get number of bedrooms and number of accommodates from user
    room_input = 3
    accommodates_input = 3
    price_range_input= "16-10000"
    exact = True

    #Data Cleaning
    amenities_data_clean = clean_amenities_data(amenities_data, amenities_required)
    listings_data_clean = clean_listings_data(listings_data, accommodates_input, room_input, price_range_input, exact)

    #add a column for number of amenities nearby to each listing
    listings_data_clean['num_amenities_nearby'] = listings_data_clean.apply(lambda x: num_amenities(x['latitude'], x['longitude'], amenities_data_clean), axis = 1)


    # sort by price in ascending order
    listings_by_price = listings_data_clean.sort_values(by="price", ascending=True).reset_index(drop=True)

    # sort by review_scores_values in descending order, drop na scores(?)
    listings_by_rscore = listings_data_clean.sort_values(['review_scores_value'], ascending=False).dropna().reset_index(drop=True)

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

    sorted_output = pd.concat([x, y], axis=0, ignore_index=True)
    print(sorted_output)
    # temp_output.shape
    # temp_output

    # display amenities distance/types around the selected listing 
    # for the first option example:
    amen_output = amenities_data_clean.copy()
    lon = sorted_output.iloc[0,7] # lon from utput listing 
    lat = sorted_output.iloc[0,6] # lat

    distance = haversine_distance(amen_output, lon , lat)
    amen_output['distance'] = distance
    data_withinR = amen_output.loc[amenities_data_clean['distance'] < 1000].reset_index(drop=True)
    print(data_withinR)
    

if __name__ == "__main__":
    main()
