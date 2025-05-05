import pandas as pd
import ast
import pycountry_convert as pc

def process_continent_features(df):
    # Define continent columns
    continent_columns = ['Asia', 'Africa', 'Europe', 'North America', 'South America', 'Australia', 'Continent Unknown']
    
    # Helper: Clean region/country lists
    def clean_region_string(region_str):
        try:
            region_list = ast.literal_eval(region_str)
            return [r for r in region_list if r != '\\N']
        except:
            return []

    # Helper: Get continent from country code
    def get_continent(country_code):
        try:
            country_alpha2 = country_code.upper()
            continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
            continent_map = {
                'AS': 'Asia',
                'AF': 'Africa',
                'EU': 'Europe',
                'NA': 'North America',
                'SA': 'South America',
                'OC': 'Australia'
            }
            return continent_map.get(continent_code, 'Continent Unknown')
        except:
            return 'Continent Unknown'

    # Clean and combine countries
    df['regions'] = df['regions'].apply(clean_region_string)
    df['countryOfOrigin'] = df['countryOfOrigin'].apply(clean_region_string)
    df['all_countries'] = df.apply(lambda row: list(set(row['regions']) | set(row['countryOfOrigin'])), axis=1)

    # Compute continent frequencies
    def frequency_encode_continents(row):
        freq_dict = {continent: 0 for continent in continent_columns}
        for code in row['all_countries']:
            continent = get_continent(code)
            freq_dict[continent] += 1
        return pd.Series(freq_dict)

    # Apply and concat
    continent_freq_df = df.apply(frequency_encode_continents, axis=1)
    df = pd.concat([df, continent_freq_df], axis=1)

    # Drop intermediate columns
    df.drop(['regions', 'countryOfOrigin', 'all_countries'], axis=1, inplace=True)
    
    return df
