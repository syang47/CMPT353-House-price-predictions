import pandas as pd
import numpy as np


#read data
listings_data = pd.read_csv('listings.csv')
amenities_data = pd.read_json('amenities-vancouver.json.gz', lines=True)

#find unique amenities and the number of them to choose which are important for a traveller
print(amenities_data['amenity'].value_counts())

# choose important amenities to work with
amenities_required = ['restaurent', 'fast_food', 'cafe','bank','atm','pharmacy','bicycle_rental','fuel','pub','bar','car_sharing','car_rental','clinic','doctors','hospital','ice_cream','fountain','theatre','police','bus_station']


#filter the amenities dataframe with only required amenities
#adapted from : https://www.kite.com/python/answers/how-to-filter-a-pandas-dataframe-with-a-list-by-%60in%60-or-%60not-in%60-in-python
bool_series = amenities_data.amenity.isin(amenities_required)
filtered_amenities_df = amenities_data[bool_series]
filtered_amenities_df.reset_index(inplace=True, drop=True)

print(filtered_amenities_df)

