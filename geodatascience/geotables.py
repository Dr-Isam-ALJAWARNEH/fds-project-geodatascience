"""
To extend the abstraction of converting a CSV into a table to handle geospatial data, we can create a new class GeoTable that inherits from the existing Table class. This new class will include functionality to process geospatial data, such as converting pairs of longitude and latitude into geometry objects, similar to how GeoDataFrames work in the geopandas library.

Below is an implementation of the GeoTable class that extends the Table class to handle geospatial data:
"""

import geohash2
import geopandas as gpd
from shapely.geometry import Point
from datascience import Table
import pandas as pd
import re

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
    def from_csv(cls, filepath_or_buffer, lon_col, lat_col, crs="EPSG:4326", geohash_precision=7, *args, **kwargs):
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
        try:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(filepath_or_buffer, *args, **kwargs)

            # Check if required columns exist
            if lon_col not in df.columns:
                raise KeyError(f"Column '{lon_col}' for longitude not found in CSV file.")
            if lat_col not in df.columns:
                raise KeyError(f"Column '{lat_col}' for latitude not found in CSV file.")

            # Check for valid longitude and latitude values (ensure they are numeric)
            if not pd.to_numeric(df[lon_col], errors='coerce').notnull().all():
                raise ValueError(f"Column '{lon_col}' contains non-numeric values or invalid data.")
            if not pd.to_numeric(df[lat_col], errors='coerce').notnull().all():
                raise ValueError(f"Column '{lat_col}' contains non-numeric values or invalid data.")

            # Convert longitude and latitude columns into a geometry column
            geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]

            # Add geohash column
            df['geohash'] = df.apply(lambda row: geohash2.encode(row[lat_col], row[lon_col], precision=geohash_precision), axis=1)

            # Create a GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)

            # Convert the GeoDataFrame into a GeoTable
            geo_table = cls.from_df(gdf)

            # Store the geometry and geohash column names
            geo_table._geometry = "geometry"
            geo_table._geohash = "geohash"

            return geo_table

        except FileNotFoundError:
            print(f"Error: The file at '{filepath_or_buffer}' was not found.")
        except KeyError as e:
            print(f"Error: {e}")
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def add_geohash(self, lon_col, lat_col, precision=7):
        """
        Adds a geohash column to the GeoTable based on latitude and longitude columns.
        
        Args:
            lon_col (str): The name of the longitude column.
            lat_col (str): The name of the latitude column.
            precision (int): The precision of the geohash (default is 7 characters).
        """
        self.append_column('geohash', [
            geohash2.encode(self.column(lat_col)[i], self.column(lon_col)[i], precision=precision)
            for i in range(self.num_rows)
        ])


    @classmethod
    def read_geojson(cls, filepath_or_buffer, *args, **kwargs):
        """
        Reads a GeoJSON file using geopandas.read_file and converts it into a GeoTable.
        
        Args:
            filepath_or_buffer (str): Path to the GeoJSON file.
            *args, **kwargs: Additional arguments passed to geopandas.read_file.
        
        Returns:
            GeoTable: A GeoTable instance with geometry data from the GeoJSON file.
        """
        try:
            # Read the GeoJSON file into a GeoDataFrame
            gdf = gpd.read_file(filepath_or_buffer, *args, **kwargs)

            # Convert the GeoDataFrame into a GeoTable
            geo_table = cls.from_df(gdf)

            # Store the geometry column name
            geo_table._geometry = "geometry"

            return geo_table

        except FileNotFoundError:
            print(f"Error: The file at '{filepath_or_buffer}' was not found.")
        except Exception as e:
            print(f"An error occurred while reading the GeoJSON file: {e}")
            

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


    def set_lat_lon_columns(self, lat_name=None, lon_name=None):
        """
        Manually sets custom column names to be used for latitude and longitude.

        This allows users to specify which columns in the table represent geographic
        coordinates, even if the column names differ from common conventions (e.g.,
        using names in other languages or domain-specific terms).

        Once set, these names will take priority during any automatic detection or
        conversion (e.g., creating geometries or converting to a GeoDataFrame).

        Parameters:
            lat_name (str): The name of the column representing latitude.
            lon_name (str): The name of the column representing longitude.

        Example:
            geo.set_lat_lon_columns("šířka", "délka")
        """
        self._custom_lat_lon['lat'] = lat_name
        self._custom_lat_lon['lon'] = lon_name

    
    def _is_lat_lon_set(self):
        """
        Checks if custom latitude and longitude column names have been set.

        Returns:
            bool: True if both 'lat' and 'lon' keys are present in _custom_lat_lon and not None, False otherwise.
        """
        return self._custom_lat_lon.get('lat') is not None and self._custom_lat_lon.get('lon') is not None
    

    def _infer_lat_lon_columns(self):
        """
        Attempts to infer the latitude and longitude column names from the table,
        using custom names (if set), flexible pattern matching, and data type heuristics.

        Returns:
            tuple: (latitude_column_name, longitude_column_name) or (None, None)
        """
        # Check for user-defined custom names
        # if self._custom_lat_lon.get('lat') in self.labels and self._custom_lat_lon.get('lon') in self.labels:
        #     return self._custom_lat_lon['lat'], self._custom_lat_lon['lon']


        possible_lat_names = ['lat', 'latitude']
        possible_lon_names = ['lon', 'lng', 'long', 'longitude']

        lat_candidates = []
        lon_candidates = []

        for col in self.labels:
            clean_col = col.lower().strip()

            # Try to find 'lat' in name using regex word boundaries or partial match
            if any(re.search(rf"\b{lat}\b", clean_col) or lat in clean_col for lat in possible_lat_names):
                lat_candidates.append(col)
            if any(re.search(rf"\b{lon}\b", clean_col) or lon in clean_col for lon in possible_lon_names):
                lon_candidates.append(col)

        def is_numeric_column(col_name):
            try:
                values = self[col_name]
                sample = [v for v in values if v is not None][:5]
                return all(isinstance(x, (float, int)) for x in sample)
            except Exception:
                return False

        # Filter further based on numeric check
        lat_col = next((c for c in lat_candidates if is_numeric_column(c)), None)
        lon_col = next((c for c in lon_candidates if is_numeric_column(c)), None)

        if not lat_col or not lon_col:
            return None, None

        # Save found names privately
        self._custom_lat_lon['lat'] = lat_col
        self._custom_lat_lon['lon'] = lon_col
        print(f"Found {lat_col} and {lon_col}")
        return lat_col, lon_col
    


    def with_columns(self, *args):
        """
        Returns a new GeoTable with additional or replaced columns.

        This method extends the default `with_columns()` from the parent `Table` class.
        After adding or replacing columns, it automatically checks for columns that
        appear to represent latitude and longitude using a private helper function
        `_infer_lat_lon_columns()`. If such columns are found, a new `geometry` column
        is generated using these coordinates and stored as `shapely.geometry.Point` objects.

        If a `geometry` column already exists, it is preserved as-is.
        If neither geometry nor recognizable latitude/longitude columns are found,
        an empty geometry column (filled with None) is created to maintain compatibility.

        Parameters:
            *args: A sequence of alternating column labels and column values,
                  similar to `datascience.Table.with_columns()`.

        Returns:
            GeoTable: A new GeoTable instance with the updated columns and a geometry column.
        """
        # Use the superclass method to get a new Table
        new_table = Table().with_columns(*args)
        
        # Ensure 'geometry' is has same number of input rows for correct insertion of other columns
        if self._geometry not in new_table.labels:
            new_table.append_column('geometry', [None] * new_table.num_rows)

        # Convert to GeoTable
        geo = GeoTable()

        # Copy columns from new_table to geo
        for label in new_table.labels:
            geo.append_column(label, new_table.column(label))

        if self._geometry in args[::2]:
            print("Using existing geometry column.")
            return geo

        lat_label, lon_label = None, None
    
        # Check if custom 'longitue' and 'latitude' column names have been used
        if self._is_lat_lon_set() and self._custom_lat_lon['lat'] in geo.labels and self._custom_lat_lon['lon'] in geo.labels:
            lat_label = self._custom_lat_lon['lat']
            lon_label = self._custom_lat_lon['lon']

        else:
            # Use private method to infer latitude and longitude columns
            lat_label, lon_label = geo._infer_lat_lon_columns()


        if lat_label and lon_label:
            # Create geometry column
            geometry = [Point(lon, lat) for lat, lon in zip(geo.column(lat_label), geo.column(lon_label))]
            geo.drop('geometry')
            geo.append_column('geometry', geometry)

            if self._is_lat_lon_set() and self._custom_lat_lon['lat'] != lat_label and self._custom_lat_lon['lon'] != lon_label:
                print(f"Replace: {lat_label} with: {self._custom_lat_lon['lat']}")
                geo.relabel(lat_label, self._custom_lat_lon['lat'])
                geo.relabel(lon_label, self._custom_lat_lon['lon'])

        return geo

def select(self, *column_labels):
    """
    Override the select method to maintain geospatial integrity.

    If latitude and longitude columns are selected (without geometry),
    automatically reconstruct the geometry column.
    """
    # Call the base Table select
    new_table = super().select(*column_labels)

    # Wrap it again into a GeoTable
    geo = GeoTable()

    for label in new_table.labels:
        geo.append_column(label, new_table.column(label))

    # Case 1: if 'geometry' was selected, no need to do anything extra
    if self._geometry in geo.labels:
        return geo

    # Case 2: Try to reconstruct geometry if latitude and longitude are present
    lat_label, lon_label = None, None

    # Use custom labels if set
    if self._is_lat_lon_set():
        lat_label = self._custom_lat_lon['lat']
        lon_label = self._custom_lat_lon['lon']

    else:
        lat_label, lon_label = self._infer_lat_lon_columns()

    if lat_label and lon_label and lat_label in geo.labels and lon_label in geo.labels:
        # Create geometry from lat/lon
        geometry = [Point(lon, lat) for lat, lon in zip(geo.column(lat_label), geo.column(lon_label))]
        geo.append_column('geometry', geometry)

    return geo
