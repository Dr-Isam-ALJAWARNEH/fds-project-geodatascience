import pandas as pd
import geopandas as gpd
from datascience import Table

class GeoTable(Table):
    def __init__(self, *args, **kwargs):
        # Call the base Table constructor
        super().__init__(*args, **kwargs)

        # Ensure 'geometry' column exists
        if 'geometry' not in self.labels:
            self.append_column('geometry', [None] * self.num_rows)

    @classmethod
    def from_csv(cls, filepath, lon_col='longitude', lat_col='latitude', geohash_precision=7):
        """
        Reads a CSV file and creates a GeoTable with geometry and geohash.

        Args:
            filepath (str): Path to the CSV file.
            lon_col (str): Name of the longitude column.
            lat_col (str): Name of the latitude column.
            geohash_precision (int): Precision level of geohash encoding (default 7).

        Returns:
            GeoTable: A new GeoTable instance.
        """
        df = pd.read_csv(filepath)

        if lon_col not in df.columns or lat_col not in df.columns:
            raise ValueError(f"Missing longitude or latitude columns: '{lon_col}', '{lat_col}'")
        
        if df[[lon_col, lat_col]].isnull().any().any():
            raise ValueError(f"Found null values in columns: '{lon_col}', '{lat_col}'")
        
        if not pd.api.types.is_numeric_dtype(df[lon_col]) or not pd.api.types.is_numeric_dtype(df[lat_col]):
            raise ValueError(f"Longitude and latitude columns must be numeric.")

        # Geometry column from lat/lon
        df['geometry'] = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]

        # Geohash encoding
        df['geohash'] = [
            geohash2.encode(lat, lon, precision=geohash_precision)
            for lat, lon in zip(df[lat_col], df[lon_col])]

        table = cls.from_df(df)
        table._geometry = 'geometry'
        return table
