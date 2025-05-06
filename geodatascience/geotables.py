"""
To extend the abstraction of converting a CSV into a table to handle geospatial data, we can create a new class GeoTable that inherits from the existing Table class. This new class will include functionality to process geospatial data, such as converting pairs of longitude and latitude into geometry objects, similar to how GeoDataFrames work in the geopandas library.

Below is an implementation of the GeoTable class that extends the Table class to handle geospatial data:
"""
""
import geohash2
import geopandas as gpd
from shapely.geometry import Point
from datascience import Table
import pandas as pd
import re
import matplotlib.pyplot as plt



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

    def distance_to(self, other=None, ref_index=0, new_column='distance_to_ref', same_table=False):
        """
        Computes distance in meters from each point in this GeoTable to a reference point,
        either from another GeoTable or from itself.

        Args:
            other (GeoTable or None): Another GeoTable containing the reference point, or None if same_table is True.
            ref_index (int): Index of the reference point in the other GeoTable or this one if same_table is True.
            new_column (str): Column name to store distances.
            same_table (bool): If True, use this table as the reference table (i.e., compare points to a point in self).
        """
        # Validate input
        if same_table:
            if ref_index < 0 or ref_index >= self.num_rows:
                raise IndexError(f"Reference index {ref_index} is out of bounds for this GeoTable.")
            ref_gdf = self.to_geodataframe().set_crs("EPSG:4326")
            gdf_self = ref_gdf
            ref_point = ref_gdf.to_crs("EPSG:3857").geometry.iloc[ref_index]
        else:
            if not isinstance(other, GeoTable):
                raise TypeError("If same_table is False, 'other' must be a GeoTable.")
            gdf_self = self.to_geodataframe().set_crs("EPSG:4326")
            ref_gdf = other.to_geodataframe().set_crs("EPSG:4326")
            if ref_index < 0 or ref_index >= len(ref_gdf):
                raise IndexError(f"Reference index {ref_index} is out of bounds for the other GeoTable.")
            ref_point = ref_gdf.to_crs("EPSG:3857").geometry.iloc[ref_index]

        # Reproject to metric CRS for distance calculation
        gdf_self_proj = gdf_self.to_crs("EPSG:3857")
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
    

    @classmethod
    def _copy_geo_state(self, target):
        """Safely copies only GeoTable-specific state to another instance"""

        # Ensure defaults exist if somehow undefined
        target._geometry = getattr(self, '_geometry', 'geometry')
        target._custom_lat_lon = getattr(self, '_custom_lat_lon', {'lat': None, 'lon': None}).copy()
        print(target)
        return target


    def _set_geo_state(self):
        """Sets GeoTable-specific state"""

        self._geometry = 'geometry' # Set geometry
        return self




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

        geo = GeoTable()._copy_geo_state(self)
        geo.append_column('geometry', [None] * new_table.num_rows)


        for label in new_table.labels:
            geo.append_column(label, new_table.column(label))

        if self._geometry in args[::2]:
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
            geo.append_column('geometry', geometry)

            if self._is_lat_lon_set() and self._custom_lat_lon['lat'] != lat_label and self._custom_lat_lon['lon'] != lon_label:
                print(f"Replace: {lat_label} with: {self._custom_lat_lon['lat']}")
                geo.relabel(lat_label, self._custom_lat_lon['lat'])
                geo.relabel(lon_label, self._custom_lat_lon['lon'])

        return geo



    def select(self, *column_labels):
        """
        Select columns from the GeoTable while preserving geospatial properties.
        
        This method mimics the behavior of GeoDataFrame.select() where:
        - Selecting ONLY the geometry column returns a GeoTable
        - Selecting geometry PLUS other columns returns a GeoTable  
        - Selecting ONLY non-geometry columns returns a regular Table
        
        Parameters
        ----------
        *column_labels : str
            One or more column names to select
            
        Returns
        -------
        GeoTable or Table
            Returns GeoTable if geometry column is selected (alone or with others),
            otherwise returns a regular Table
            
        Notes
        -----
        Behavior matches GeoPandas GeoDataFrame:
        - Dropping all geometry columns converts to regular DataFrame
        - Keeping geometry column maintains geospatial properties
        
        Examples
        --------
        >>> gt = GeoTable().with_columns(
        ...     'City', ['Paris', 'Berlin'],
        ...     'Latitude', [48.8566, 52.52],
        ...     'Longitude', [2.3522, 13.405]
        ... )
        
        # Returns GeoTable (geometry selected)
        >>> geo = gt.select('geometry')  
        
        # Returns GeoTable (geometry + others)
        >>> geo_city = gt.select('geometry', 'City')
        
        # Returns regular Table (no geometry)
        >>> cities = gt.select('City')  
        """
        # Convert to list for easier manipulation
        columns = list(column_labels)
        
        # Case 1: Selecting ONLY the geometry column
        if columns == [self._geometry]:
            
            geo = GeoTable()
            geo._set_geo_state()
            geo.append_column(self._geometry, self.column(self._geometry))
            return geo
        
        # Case 2: Selecting geometry PLUS other columns
        elif self._geometry in columns:

            geo = GeoTable()
            geo._set_geo_state()
            for label in columns:
                geo.append_column(label, self.column(label))
            return geo
        
        # Case 3: Selecting non-geometry columns
        else:
            if len(columns) == 1 and isinstance(columns[0], (tuple, list)):
                columns = columns[0]

            return Table().with_columns(*[(label, self.column(label)) for label in columns])



    def drop(self, *column_labels):
        """
        Drop columns from the GeoTable while handling geospatial properties.
        
        Mimics GeoPandas behavior where:
        - Dropping the geometry column returns a regular Table
        - Keeping the geometry column returns a GeoTable
        
        Parameters
        ----------
        *column_labels : str
            One or more column names to drop
            
        Returns
        -------
        Table or GeoTable
            Returns a regular Table if geometry column is dropped,
            otherwise returns a GeoTable with geospatial capabilities
            
        Examples
        --------
        >>> gt = GeoTable().with_columns(
        ...     'City', ['Paris', 'Berlin'],
        ...     'Latitude', [48.8566, 52.52],
        ...     'Longitude', [2.3522, 13.405]
        ... )
        
        # Returns regular Table (geometry dropped)
        >>> no_geo = gt.drop('geometry')  
        
        # Returns GeoTable (keeping geometry)
        >>> no_city = gt.drop('City')  
        
        # Returns regular Table (geometry explicitly dropped)
        >>> no_geo = gt.drop('geometry', 'City')  
        """
        # Convert to list for easier manipulation
        drop_cols = list(column_labels)
        
        # Case 1: Selecting ONLY the geometry column
        if self._geometry in drop_cols:
            dropped = Table().with_columns(*[(label, self.column(label)) for label in self.labels if label not in drop_cols])
            return dropped
        
        # Convert to GeoTable and copy state
        geo = GeoTable()
        geo._set_geo_state()
        
        # Add remaining columns
        for label in self.labels:
            if label not in drop_cols:
              geo.append_column(label, self.column(label))
            
        return geo

    

    def where(self, column_or_predicate, value=None):
        """
        Enhanced where() with geospatial error handling.
        """
        try:
            return super().where(column_or_predicate, value)
        except AttributeError as e:
            if 'within' in str(e) or 'intersects' in str(e):
                raise AttributeError(
                    f"Column '{column_or_predicate}' must contain Shapely geometries. "
                    "Ensure you:\n"
                    "1. Created the table using from_geojson()/from_csv()\n"
                    "2. Didn't accidentally drop the geometry column"
                ) from e
            raise  # Re-raises the original AttributeError unchanged



    def spatial_join(self, other, how='inner', predicate='intersects'):
        """
        Perform a spatial join between this GeoTable and another GeoTable.

        Args:
            other (GeoTable): Another GeoTable to join with.
            how (str): Type of join: 'left', 'right', 'inner' (default: 'inner').
            predicate (str): Spatial predicate: 'intersects', 'within', 'contains', etc. (default: 'intersects').

        Returns:
            GeoTable: Result of the spatial join as a new GeoTable.
        """
        if not isinstance(other, GeoTable):
            raise TypeError("The 'other' argument must be a GeoTable.")

        gdf_self = self.to_geodataframe().copy()
        gdf_other = other.to_geodataframe().copy()

        # Ensure both are in the same CRS
        gdf_self = gdf_self.set_crs("EPSG:4326")
        gdf_other = gdf_other.set_crs("EPSG:4326")

        try:
            joined = gpd.sjoin(gdf_self, gdf_other, how=how, predicate=predicate)

            # Clean up index columns if present
            joined.reset_index(drop=True, inplace=True)
            if "index_right" in joined.columns:
                joined.drop(columns=["index_right"], inplace=True)

            return GeoTable.from_df(joined)

        except Exception as e:
            print(f"Spatial join failed: {e}")
            return None



    def plot_sjoined_interactive(self, neighbor_col='geohash', zoom=12):
        """
        Plots GeoTable points, grouping by the same 'neighbor' in the same color.

        Args:
            neighbor_col (str): Column name indicating neighborhood/grouping (e.g., 'geohash' or cluster label).
            zoom (int): Zoom level for the base map.
        """
        if self._geometry not in self.labels:
            raise ValueError(f"Geometry column '{self._geometry}' not found.")

        if neighbor_col not in self.labels:
            raise ValueError(f"Neighbor column '{neighbor_col}' not found.")

        # Convert to GeoDataFrame
        gdf = self.to_geodataframe().set_crs("EPSG:4326")
        gdf = gdf.to_crs("EPSG:3857")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        gdf.plot(ax=ax, column=neighbor_col, cmap='tab20', legend=True,
                markersize=40, alpha=0.8, edgecolor='black')

        # Add basemap
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=zoom)

        ax.set_axis_off()
        plt.tight_layout()
        plt.title(f"Points colored by '{neighbor_col}'")
        plt.show()



    def show(self, max_rows=10):
        """
        Show the GeoTable in terminal or notebook.
        Falls back to plain text in terminal.
        """
        import sys

        if 'ipykernel' in sys.modules:
            # We're in Jupyter
            # from IPython.display import display, HTML
            # display(HTML(super()._repr_html_()))
            super().show()
            
        else:
            # Terminal fallback
            print("\n".join([
                "\t".join(self.labels)
            ] + [
                "\t".join(str(v) for v in self.row(i))
                for i in range(min(self.num_rows, max_rows))
            ]))

    def spatial_groupby_mean(self, regions_table, on='geometry', features=None, agg_name_prefix='mean_', predicate='intersects'):
        """
        Performs a spatial join and groups point features by region geometries to compute mean values.
        
        Args:
            regions_table (GeoTable): Polygon GeoTable to group by.
            on (str): Geometry column to group by (default: 'geometry').
            features (list): List of feature columns in self to compute mean on.
            agg_name_prefix (str): Prefix for new aggregated columns.
            predicate (str): Spatial predicate for join (default: 'intersects').
        
        Returns:
            GeoTable: The regions_table with new mean feature columns.
        """
        joined = self.spatial_join(regions_table, how='inner', predicate=predicate)
        gdf = joined.to_geodataframe()

        if features is None:
            features = gdf.select_dtypes(include=['float', 'int']).columns.drop('geometry', errors='ignore').tolist()

        grouped = gdf.groupby(gdf.index)[features].mean().add_prefix(agg_name_prefix)
        
        result_gdf = regions_table.to_geodataframe().join(grouped, how='left')
        return GeoTable.from_df(result_gdf)

    def spatial_correlation_matrix(self, method='pearson'):
        """
        Computes correlation matrix for all numeric columns, excluding geometry.
        
        Args:
            method (str): Correlation method ('pearson', 'spearman', 'kendall').
        
        Returns:
            pd.DataFrame: Correlation matrix.
        """
        df = self.to_df().copy()
        df_numeric = df.select_dtypes(include=['float', 'int']).drop(columns=['geometry'], errors='ignore')
        return df_numeric.corr(method=method)
    def plot_correlation_heatmap(self, method='pearson', figsize=(10, 8), cmap='coolwarm', annot=True):
        """
        Plots a heatmap of the correlation matrix for numeric columns.
        
        Args:
            method (str): Correlation method.
            figsize (tuple): Figure size.
            cmap (str): Colormap.
            annot (bool): Whether to annotate cells.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        corr = self.spatial_correlation_matrix(method)
        plt.figure(figsize=figsize)
        sns.heatmap(corr, cmap=cmap, annot=annot, fmt=".2f", square=True)
        plt.title(f'{method.title()} Correlation Heatmap')
        plt.show()

    def plot_spatial_scatter(self, x_col, y_col, label_points=False):
        """
        Plots a scatter plot between two numeric columns in the GeoTable.

        Args:
            x_col (str): Name of the x-axis feature (e.g., 'mean_pm25').
            y_col (str): Name of the y-axis feature (e.g., 'mean_humidity').
            label_points (bool): Whether to label each point with its index or name.
        """
        import matplotlib.pyplot as plt

        df = self.to_df()

        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Columns '{x_col}' and/or '{y_col}' not found.")

        x = df[x_col]
        y = df[y_col]

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color='blue', alpha=0.6, edgecolor='k')

        if label_points:
            for i, (xi, yi) in enumerate(zip(x, y)):
                plt.text(xi, yi, str(i), fontsize=8, alpha=0.7)

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Scatter Plot: {y_col} vs. {x_col}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    def plot_spatial_regression(self, x_col, y_col, label_points=False):
        """
        Plots a scatter plot and regression line between two numeric columns.

        Args:
            x_col (str): Feature name on x-axis.
            y_col (str): Feature name on y-axis.
            label_points (bool): Optionally show region index as label.
        """
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LinearRegression
        import numpy as np

        df = self.to_df().copy()

        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Columns '{x_col}' and/or '{y_col}' not found in table.")

        x = df[x_col].values.reshape(-1, 1)
        y = df[y_col].values.reshape(-1, 1)

        # Fit regression model
        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color='skyblue', edgecolor='black', label='Data')
        plt.plot(x, y_pred, color='green', linewidth=2, label='Regression Line')

        if label_points:
            for i, (xi, yi) in enumerate(zip(x.flatten(), y.flatten())):
                plt.text(xi, yi, str(i), fontsize=8)

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Regression: {y_col} vs {x_col}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Optional: print model coefficients
        slope = model.coef_[0][0]
        intercept = model.intercept_[0]
        print(f"Regression Line: {y_col} = {slope:.4f} * {x_col} + {intercept:.4f}")

    def plot_manual_least_squares_regression(self, x_col, y_col, label_points=False):
        """
        Manually computes and plots the least squares regression line (no sklearn).

        Args:
            x_col (str): Feature name on x-axis.
            y_col (str): Feature name on y-axis.
            label_points (bool): Optionally annotate each point.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        df = self.to_df()

        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Columns '{x_col}' and/or '{y_col}' not found.")

        x = np.array(df[x_col])
        y = np.array(df[y_col])

        # Manual least squares
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        m = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
        b = y_mean - m * x_mean

        y_pred = m * x + b

        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color='skyblue', edgecolor='black', label='Data')
        plt.plot(x, y_pred, color='blue', label='Least Squares Line')

        if label_points:
            for i, (xi, yi) in enumerate(zip(x, y)):
                plt.text(xi, yi, str(i), fontsize=8)

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Manual Least Squares: {y_col} vs {x_col}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print(f"Manual Regression Line: {y_col} = {m:.4f} * {x_col} + {b:.4f}")

    def plot_regression_diagnostics(self, x_col, y_col):
        """
        Plots standard linear regression diagnostics for two numeric features.

        Args:
            x_col (str): Independent variable.
            y_col (str): Dependent variable.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as stats
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        df = self.to_df()

        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Columns '{x_col}' and/or '{y_col}' not found.")

        x = df[x_col].values.reshape(-1, 1)
        y = df[y_col].values

        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)
        residuals = y - y_pred
        r2 = r2_score(y, y_pred)

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Scatter + regression
        axs[0, 0].scatter(x, y, color='skyblue', edgecolor='black', label='Data')
        axs[0, 0].plot(x, y_pred, color='red', linewidth=2, label='Regression Line')
        axs[0, 0].set_title(f"Regression Line\nR² = {r2:.4f}")
        axs[0, 0].set_xlabel(x_col)
        axs[0, 0].set_ylabel(y_col)
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Residuals vs Fitted
        axs[0, 1].scatter(y_pred, residuals, edgecolor='black', alpha=0.6)
        axs[0, 1].axhline(0, color='red', linestyle='--')
        axs[0, 1].set_title("Residuals vs Fitted")
        axs[0, 1].set_xlabel("Predicted")
        axs[0, 1].set_ylabel("Residuals")
        axs[0, 1].grid(True)

        # Histogram of residuals
        axs[1, 0].hist(residuals, bins=20, color='purple', alpha=0.7)
        axs[1, 0].set_title("Histogram of Residuals")
        axs[1, 0].set_xlabel("Residual")
        axs[1, 0].set_ylabel("Frequency")

        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axs[1, 1])
        axs[1, 1].set_title("Q-Q Plot")

        plt.tight_layout()
        plt.show()

    def sample(self, k=None, with_replacement=True, weights=None):
        """
        Draw a random sample of rows from the GeoTable.

        Args:
            k (int): Number of rows to sample. If None, sample the same number as the table's rows.
            with_replacement (bool): Whether to sample with replacement (default: True).
            weights (array-like): Probabilities for each row (default: None for equal probability).

        Returns:
            GeoTable: A new GeoTable with sampled rows, preserving geospatial properties.

        Example:
            >>> gt = GeoTable.from_csv('NYC_PM.csv', lon_col='longitude', lat_col='latitude')
            >>> sampled = gt.sample(100)  # Sample 100 rows with replacement
        """
        # Perform sampling using the parent Table's sample method
        sampled_table = super().sample(k, with_replacement=with_replacement, weights=weights)

        # Create a new GeoTable and copy geospatial state
        geo = GeoTable()
        geo = self._copy_geo_state(geo)

        # Copy all columns from sampled_table to geo
        for label in sampled_table.labels:
            geo.append_column(label, sampled_table.column(label))

        return geo
    

    def stratified_sample(self, strata_column, k=None, sizes=None, with_replacement=True):
        """
        Perform stratified sampling based on a column.

        Args:
            strata_column (str): Column name to define strata (e.g., 'geohash', 'city').
            k (int): Total number of rows to sample (proportional across strata). Ignored if sizes is provided.
            sizes (dict): Dictionary mapping stratum values to sample sizes (e.g., {'gh1': 10, 'gh2': 20}).
            with_replacement (bool): Whether to sample with replacement (default: True).

        Returns:
            GeoTable: A new GeoTable with sampled rows, preserving geospatial properties.

        Example:
            >>> gt = GeoTable.from_csv('NYC_PM.csv', lon_col='longitude', lat_col='latitude')
            >>> stratified = gt.stratified_sample('geohash', k=100)  # Sample 100 rows proportionally
            >>> stratified = gt.stratified_sample('geohash', sizes={'gh1': 50, 'gh2': 30})
        """
        if strata_column not in self.labels:
            raise ValueError(f"Strata column '{strata_column}' not found in GeoTable.")

        # Group by strata_column to get counts and values
        grouped = self.group(strata_column)
        strata_values = grouped.column(strata_column)
        strata_counts = grouped.column('count')
        total_rows = self.num_rows

        # Compute sample sizes
        if sizes:
            sample_sizes = sizes
            # Validate sizes
            for stratum, size in sample_sizes.items():
                if stratum not in strata_values:
                    raise ValueError(f"Stratum '{stratum}' not found in column '{strata_column}'.")
                if not with_replacement:
                    stratum_count = strata_counts[list(strata_values).index(stratum)]
                    if size > stratum_count:
                        raise ValueError(
                            f"Sample size {size} for stratum '{stratum}' exceeds available rows "
                            f"({stratum_count}) with with_replacement=False."
                        )
        else:
            if k is None:
                k = total_rows
            proportions = strata_counts / total_rows
            sample_sizes = {val: max(1, int(np.round(prop * k))) for val, prop in zip(strata_values, proportions)}

        # Collect sampled rows as a list of dictionaries
        sampled_rows = []
        for stratum, size in sample_sizes.items():
            if size > 0:
                stratum_table = self.where(strata_column, stratum)
                if stratum_table.num_rows == 0:
                    continue
                sampled = stratum_table.sample(k=size, with_replacement=with_replacement)
                for i in range(sampled.num_rows):
                    row_dict = {label: sampled.column(label)[i] for label in sampled.labels}
                    sampled_rows.append(row_dict)

        # If no rows were sampled, return an empty GeoTable
        if not sampled_rows:
            result = GeoTable()
            result = self._copy_geo_state(result)
            for label in self.labels:
                result.append_column(label, [])
            return result

        # Create result GeoTable
        result = GeoTable()
        result = self._copy_geo_state(result)

        # Convert sampled rows to columns
        all_labels = set()
        for row in sampled_rows:
            all_labels.update(row.keys())

        for label in all_labels:
            column_data = [row.get(label, None) for row in sampled_rows]
            result.append_column(label, column_data)

        return result
    
    def empirical_distribution(self, statistic, n_samples=1000, sample_size=None, with_replacement=True, column=None):
        """
        Generate an empirical distribution of a statistic by repeated sampling.

        Args:
            statistic (callable): Function that takes a GeoTable and returns a numeric value
                                (e.g., lambda t: np.mean(t.column('distance_to_ref'))).
            n_samples (int): Number of times to sample and compute the statistic (default: 1000).
            sample_size (int): Number of rows to sample each time. If None, use table size.
            with_replacement (bool): Whether to sample with replacement (default: True).
            column (str): Optional column name to pass to statistic (for validation).

        Returns:
            GeoTable: A GeoTable with one column ('statistic') containing the computed values.

        Example:
            >>> gt = GeoTable.from_csv('NYC_PM.csv', lon_col='longitude', lat_col='latitude')
            >>> gt.distance_to(ref_index=0, new_column='distance_to_ref')
            >>> dist = gt.empirical_distribution(
            ...     statistic=lambda t: np.mean(t.column('distance_to_ref')),
            ...     n_samples=1000,
            ...     sample_size=100
            ... )
            >>> dist.hist('statistic')
        """
        if column and column not in self.labels:
            raise ValueError(f"Column '{column}' not found in GeoTable.")

        if sample_size is None:
            sample_size = self.num_rows

        # Perform repeated sampling and compute statistic
        stats = []
        for _ in range(n_samples):
            sample = self.sample(sample_size, with_replacement=with_replacement)
            stat_value = statistic(sample)
            stats.append(stat_value)

        # Create a GeoTable to store results
        result = GeoTable()
        result.append_column('statistic', stats)

        return result

        

    



    
