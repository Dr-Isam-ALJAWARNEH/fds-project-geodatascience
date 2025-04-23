"""
To extend the abstraction of converting a CSV into a table to handle geospatial data, we can create a new class GeoTable that inherits from the existing Table class. This new class will include functionality to process geospatial data, such as converting pairs of longitude and latitude into geometry objects, similar to how GeoDataFrames work in the geopandas library.

Below is an implementation of the GeoTable class that extends the Table class to handle geospatial data:
"""

import geopandas as gpd
from shapely.geometry import Point
from datascience import Table
import pandas as pd

class GeoTable(Table):
    """
    A GeoTable is an extension of the Table class that supports geospatial data.
    It allows for reading CSV files containing longitude and latitude columns,
    converting them into geometry objects, and performing geospatial operations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'geometry' not in self.labels:
            self.append_column('geometry', [None] * self.num_rows)
        self._geometry = 'geometry'
        self._custom_lat_lon = {
            'lat': None,
            'lon': None
        }

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
        df = pd.read_csv(filepath_or_buffer, *args, **kwargs)

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

    def distance_to(self, other, ref_index=0, new_column='distance_to_ref'):
        """
        Computes distance in meters from each point in this GeoTable to a reference point
        in another GeoTable.

        Args:
            other (GeoTable): Another GeoTable containing reference point.
            ref_index (int): Index of the reference point in the other GeoTable.
            new_column (str): Column name to store distances.
        """
        if not isinstance(other, GeoTable):
            raise TypeError("Other must be a GeoTable.")

        gdf_self = self.to_geodataframe().set_crs("EPSG:4326")
        gdf_other = other.to_geodataframe().set_crs("EPSG:4326")

        if ref_index < 0 or ref_index >= len(gdf_other):
            raise IndexError(f"Reference index {ref_index} out of bounds.")

        # Reproject to metric CRS for distance calculation
        gdf_self_proj = gdf_self.to_crs("EPSG:3857")
        ref_point = gdf_other.to_crs("EPSG:3857").geometry.iloc[ref_index]

        distances = gdf_self_proj.geometry.distance(ref_point)
        self[new_column] = distances.round(1).tolist()

    def plot(self, zoom=12, **kwargs):
        """
        Plot the GeoTable on a basemap using contextily.

        Args:
            zoom (int): Zoom level for the basemap.
            **kwargs: Additional arguments passed to GeoDataFrame.plot()
        """
        if self._geometry not in self.labels:
            raise ValueError(f"Geometry column '{self._geometry}' not found.")

        try:
            import contextily as ctx
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Install contextily and matplotlib to enable plotting.")

        # Convert to GeoDataFrame with proper CRS
        gdf = self.to_geodataframe().set_crs("EPSG:4326")
        gdf_proj = gdf.to_crs("EPSG:3857")  # Web Mercator projection

        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the points on top of basemap
        gdf_proj.plot(ax=ax, markersize=20, color='blue', alpha=0.7, edgecolor='white', linewidth=0.5)

        # Add basemap below the points
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=zoom)

        # Clean up plot aesthetics
        ax.set_axis_off()
        plt.tight_layout()

        # Display the figure (no return to suppress extra output)
        plt.show()


    def to_geodf(self, lat_col='latitude', lon_col='longitude', crs='EPSG:4326'):
        """
        Convert the GeoTable into a GeoDataFrame using either the existing 'geometry' column,
        or the specified latitude and longitude columns to create geometry.

        Parameters:
        - lat_col (str): Name of the latitude column (optional if 'geometry' is present).
        - lon_col (str): Name of the longitude column (optional if 'geometry' is present).
        - crs (str): Coordinate Reference System for the geometry. Default is 'EPSG:4326'.

        Returns:
        - geopandas.GeoDataFrame: A geospatially-aware dataframe.
        """
        df = self.to_df()

        if self._geometry in self.labels and all(isinstance(g, Point) for g in self['geometry'] if g is not None):
            # Use existing geometry column
            geometry = self['geometry']
            print("🧪 Using existing geometry column to create GeoDataFrame.")

        else:
            raise ValueError("No geometry data available! Provide valid 'geometry' column or latitude/longitude columns.")

        # Create GeoDataFrame with geometry
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
        return gdf