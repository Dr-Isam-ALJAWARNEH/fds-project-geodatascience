"""
To extend the abstraction of converting a CSV into a table to handle geospatial data, we can create a new class GeoTable that inherits from the existing Table class. This new class will include functionality to process geospatial data, such as converting pairs of longitude and latitude into geometry objects, similar to how GeoDataFrames work in the geopandas library.

Below is an implementation of the GeoTable class that extends the Table class to handle geospatial data:
"""

import geopandas as gpd
from shapely.geometry import Point
from datascience import Table

class GeoTable(Table):
    """
    A GeoTable is an extension of the Table class that supports geospatial data.
    It allows for reading CSV files containing longitude and latitude columns,
    converting them into geometry objects, and performing geospatial operations.
    """

    def __init__(self, labels=None, formatter=None):
        super().__init__(labels, formatter)
        self._geometry = None  # To store the geometry column

    @classmethod
    def from_csv(cls, filepath_or_buffer, lon_col, lat_col, crs="EPSG:4326", *args, **kwargs):
        """
        Read a CSV file and convert it into a GeoTable with a geometry column.

        Args:
            filepath_or_buffer: Path to the CSV file or a file-like object.
            lon_col (str): Name of the column containing longitude values.
            lat_col (str): Name of the column containing latitude values.
            crs (str): Coordinate Reference System (default is EPSG:4326 for WGS84).
            *args, **kwargs: Additional arguments passed to pandas.read_csv.

        Returns:
            A GeoTable instance with a geometry column created from lon_col and lat_col.
        """
        # Read the CSV file into a pandas DataFrame
        df = pandas.read_csv(filepath_or_buffer, *args, **kwargs)

        # Convert longitude and latitude columns into a geometry column
        geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]

        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)

        # Convert the GeoDataFrame into a GeoTable
        geo_table = cls.from_df(gdf)

        # Store the geometry column name
        geo_table._geometry = "geometry"

        return geo_table

    def to_geodataframe(self):
        """
        Convert the GeoTable back into a GeoDataFrame.

        Returns:
            A GeoDataFrame representation of the GeoTable.
        """
        # Convert the GeoTable to a pandas DataFrame
        df = self.to_df()

        # Create a GeoDataFrame using the geometry column
        gdf = gpd.GeoDataFrame(df, geometry=self._geometry, crs="EPSG:4326")

        return gdf

    def plot(self, *args, **kwargs):
        """
        Plot the geospatial data in the GeoTable.

        Args:
            *args, **kwargs: Arguments passed to geopandas.GeoDataFrame.plot().
        """
        gdf = self.to_geodataframe()
        gdf.plot(*args, **kwargs)

    def distance_to(self, other_geo_table, target_col="distance"):
        """
        Compute the distance between geometries in this GeoTable and another GeoTable.

        Args:
            other_geo_table (GeoTable): Another GeoTable to compute distances to.
            target_col (str): Name of the column to store the computed distances.

        Returns:
            A new GeoTable with an additional column containing distances.
        """
        if not isinstance(other_geo_table, GeoTable):
            raise ValueError("The other table must be a GeoTable.")

        # Convert both tables to GeoDataFrames
        gdf1 = self.to_geodataframe()
        gdf2 = other_geo_table.to_geodataframe()

        # Compute distances
        distances = gdf1.distance(gdf2.iloc[0].geometry)

        # Add distances as a new column
        new_table = self.with_column(target_col, distances.values)

        return new_table