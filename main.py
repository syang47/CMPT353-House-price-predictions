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
    filtered_amenities_df.reset_index(inplace=True, drop=True)

    return filtered_amenities_df


#reference: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
def haversine_distance(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000


def clean_listings_data(listings_data):
    #keep only the columns we need
    columns_needed = ['id', 'listing_url', 'name', 'description', 'picture_url', 'latitude', 'longitude', 'property_type', 'accommodates', 'bedrooms', 'beds', 'amenities', 'price']
    listings_data = listings_data[columns_needed]
    return listings_data



#TODO: return a dictionary with number of amenities in a 1km radius of this lat and lon
def num_amenities(lat, lon, amenities_data_clean, amenities_required):
    amenities_dict = dict.fromkeys(amenities_required,0)
    


def main():
    #Read Data
    listings_data, amenities_data = read_data()
    
    # Change amenities here 
    amenities_required = ['restaurent', 'fast_food', 'cafe','bank','atm','pharmacy','bicycle_rental','fuel','pub','bar','car_sharing','car_rental','clinic','doctors','hospital','ice_cream','fountain','theatre','police','bus_station']

    #Data Cleaning
    amenities_data_clean = clean_amenities_data(amenities_data, amenities_required)
    listings_data_clean = clean_listings_data(listings_data)

    #TODO: Add a column to listings dataset:  each element with a dictionary of number of amenities in a 1km radius of a listing.




if __name__ == "__main__":
    main()
