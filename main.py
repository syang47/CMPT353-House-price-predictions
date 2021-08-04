import pandas as pd
import numpy as np
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

def clean_listings_data(listings_data, num_accomodates, num_bedrooms, max_price):
    #keep only the columns we need
    columns_needed = ['id', 'listing_url', 'name', 'description', 'picture_url', 'latitude', 'longitude', 'property_type', 'accommodates', 'bedrooms', 'beds', 'amenities', 'price']
    listings_data = listings_data[columns_needed]

    # Change price from str to float
    listings_data['price'] = listings_data['price'].apply(lambda x: float(x.replace('$','')))

    #filter based on accomodates, bedrooms and price requirements
    listings_data = listings_data.loc[(listings_data['price'] <= max_price) & (listings_data['accommodates'] >= num_accomodates) & (listings_data['bedrooms'] >= num_bedrooms)]

    return listings_data


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

    # TODO:turn this into user input in the end
    num_accomodates = 4
    num_bedrooms = 2
    max_price = 200 

    #Data Cleaning
    amenities_data_clean = clean_amenities_data(amenities_data, amenities_required)
    listings_data_clean = clean_listings_data(listings_data,num_accomodates,num_bedrooms,max_price)

    #add a column for number of amenities nearby to each listing
    listings_data_clean['num_amenities_nearby'] = listings_data_clean.apply(lambda x: num_amenities(x['latitude'], x['longitude'], amenities_data_clean), axis = 1)
    print(listings_data_clean)

if __name__ == "__main__":
    main()
