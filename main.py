import pandas as pd
import numpy as np

def read_data():
    listings_data = pd.read_csv('listings.csv.gz')
    amenities_data = pd.read_json('amenities-vancouver.json.gz', lines=True)
    return listings_data, amenities_data


def clean_amenities_data(amenities_data):

    #find unique amenities and the number of them to choose which are important for a traveller
    print(amenities_data['amenity'].value_counts())

    # choose important amenities to work with
    amenities_required = ['restaurent', 'fast_food', 'cafe','bank','atm','pharmacy','bicycle_rental','fuel','pub','bar','car_sharing','car_rental','clinic','doctors','hospital','ice_cream','fountain','theatre','police','bus_station']
    #adapted from : https://www.kite.com/python/answers/how-to-filter-a-pandas-dataframe-with-a-list-by-%60in%60-or-%60not-in%60-in-python
    bool_series = amenities_data.amenity.isin(amenities_required)
    filtered_amenities_df = amenities_data[bool_series]
    filtered_amenities_df.reset_index(inplace=True, drop=True)

    return filtered_amenities_df


def clean_listings_data(listings_data):
    #keep only the columns we need
    columns_needed = ['id', 'listing_url', 'name', 'description', 'picture_url', 'latitude', 'longitude', 'property_type', 'accommodates', 'bedrooms', 'beds', 'amenities', 'price']
    listings_data = listings_data[columns_needed]
    return listings_data

def main():
    #Read Data
    listings_data, amenities_data = read_data()
    
    #Data Cleaning
    amenities_data_clean = clean_amenities_data(amenities_data)
    listings_data_clean = clean_listings_data(listings_data)


if __name__ == "__main__":
    main()
