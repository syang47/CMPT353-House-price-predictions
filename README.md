# CMPT 353 - OSM and Airbnb Listing

The _listings_by_amenities.py_ is a python program that takes an input text file with user's Airbnb filtering preference in the following format:

accommodates: x  
bedrooms: y  
price range: z  
exact: bool  

In the above example, 'x' and 'y' will be integer values indicating the user's desired number of accommodates and bedrooms for choosing the Airbnb listings. The 'z' value will be a price range in the format of min-max, where min is the minimum price that has to be larger than 0, and max is the maximum desired price. 'bool' is the boolean value of true or false for the exact parameter. True for exact's bool means user wants to search listings with the exact preference they set, while False for exact's bool will be searching the listings with the set preference as minimum requirements. Anything strings before the colon has to stay the same in order to successfully filter base on preference.

Please refer to the sample input files in there are any confusions in setting the parameters. 

The two output files ("AirBnb_search_results.csv", "ML_Price_Prediciton.csv") will be outputted to the same folder upon success program run.

### Command to run the program with example inputs:

1. To find best Airbnb listings based on minimum preference requirements (exact = False):
    * python3 listings_by_amenities.py input1.txt    


2. To find best Airbnb listings based on exact preference requirements (exact = True):
    * python3 listings_by_amenities.py input2.txt  
