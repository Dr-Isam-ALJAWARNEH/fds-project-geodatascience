

# proposed update 1
- To extend the abstraction of converting a CSV into a table to handle geospatial data, we can create a new class GeoTable that inherits from the existing Table class. This new class will include functionality to process geospatial data, such as converting pairs of longitude and latitude into geometry objects, similar to how GeoDataFrames work in the geopandas library.

- Explanation of Key Features
1. Initialization (`__init__`) :
The `GeoTable` class inherits from the `Table` class and adds a _geometry attribute to store the name of the `geometry` column.
2. Reading CSV with Geospatial Data (`from_csv`) :
This method reads a CSV file using `pandas.read_csv`.
It extracts the longitude and latitude columns and converts them into `shapely.geometry.Point` objects.
A GeoDataFrame is created using these points and then converted into a `GeoTable`.
3. Conversion to GeoDataFrame (`to_geodataframe`) :
This method converts the `GeoTable` back into a `GeoDataFrame`, allowing users to leverage the full functionality of `geopandas`.
4. Plotting (`plot`) :
The `plot` method uses `geopandas.GeoDataFrame.plot()` to visualize the geospatial data.
Distance Calculation (`distance_to`) :
This method computes the distance between geometries in two `GeoTable` instances.
The result is added as a new column in the `GeoTable`.

Example Usage
Input CSV File (`locations.csv`):

```
name,longitude,latitude
LocationA,-122.4194,37.7749
LocationB,-118.2437,34.0522
LocationC,-73.9352,40.7306
```

Output
1. GeoTable :
```
name      | longitude | latitude | geometry
LocationA | -122.4194 | 37.7749  | POINT (-122.4194 37.7749)
LocationB | -118.2437 | 34.0522  | POINT (-118.2437 34.0522)
LocationC | -73.9352  | 40.7306  | POINT (-73.9352 40.7306)
```

2. Plot :
- A map showing the locations as points.

This implementation provides a seamless way to handle geospatial data within the Table abstraction while leveraging the power of `geopandas`.

# proposed update 2

 - Also, this abstraction should be able to convert longitude latitude pairs to geohash
    > To extend the GeoTable abstraction to include the conversion of longitude and latitude pairs into geohashes, we can integrate the geohash library. Geohash is a system that encodes geographic coordinates (latitude and longitude) into a short string of letters and digits. This encoding is useful for proximity searches and spatial indexing.

Key Features that can be Added
1. Geohash Conversion (`add_geohash`) :
This method adds a geohash column to the `GeoTable` based on the specified longitude and latitude columns.
The `precision` parameter controls the length of the geohash string, which determines the spatial resolution.
2. Automatic Geohash Column in `from_csv` :
When reading a CSV file, the `from_csv` method automatically generates a geohash column alongside the geometry column.
3. Integration with GeoDataFrame :
The `to_geodataframe` method ensures compatibility with `geopandas`, allowing users to leverage its full functionality.
Example Usage
Input CSV File (`locations.csv`):

Output
1. GeoTable
```
name      | longitude | latitude | geometry                  | geohash
LocationA | -122.4194 | 37.7749  | POINT (-122.4194 37.7749) | 9q8yyz8
LocationB | -118.2437 | 34.0522  | POINT (-118.2437 34.0522) | 9q5csmw
LocationC | -73.9352  | 40.7306  | POINT (-73.9352 40.7306)  | dr5ru7g
```

2. Plot :
A map showing the locations as points.
3. Distances :

```
name      | longitude | latitude | geometry                  | geohash  | distance_to_ref
LocationA | -122.4194 | 37.7749  | POINT (-122.4194 37.7749) | 9q8yyz8 | 2224.5
LocationB | -118.2437 | 34.0522  | POINT (-118.2437 34.0522) | 9q5csmw | 1892.3
LocationC | -73.9352  | 40.7306  | POINT (-73.9352 40.7306)  | dr5ru7g | 2567.8
```
4. Manually Added Geohash :
```
name      | longitude | latitude | geometry                  | geohash  | geohash_manual
LocationA | -122.4194 | 37.7749  | POINT (-122.4194 37.7749) | 9q8yyz8 | 9q8yyz8
LocationB | -118.2437 | 34.0522  | POINT (-118.2437 34.0522) | 9q5csmw | 9q5csmw
LocationC | -73.9352  | 40.7306  | POINT (-73.9352 40.7306)  | dr5ru7g | dr5ru7g
```
To extend the `GeoTable` abstraction to include the conversion of longitude and latitude pairs into geohashes, we can integrate the `geohash` library. Geohash is a system that encodes geographic coordinates (latitude and longitude) into a short string of letters and digits. This encoding is useful for proximity searches and spatial indexing.



- This implementation provides seamless handling of geospatial data, including geometry creation and geohash encoding, while maintaining compatibility with the `Table` abstraction.

# proposed update 2

- Include a method to read geojson file, then join csv based on geometry point with polygons in geojson file, probably using sjoin method, but the new join method should make it easier and straightforward to endusers.

- To extend the `GeoTable` class to include a method for reading GeoJSON files and performing spatial joins with CSV data, we can integrate the `geopandas` library's `sjoin` functionality. The goal is to make this process straightforward for end users by abstracting away the complexity of spatial joins.

an updated implementation of the `GeoTable` class that includes:

A method to read GeoJSON files (`read_geojson`).
A simplified spatial join method (`spatial_join`) that uses `sjoin` internally but provides a user-friendly interface.

Key Features can be Added
1. Reading GeoJSON Files (read_geojson) :
This method reads a GeoJSON file using geopandas.read_file and converts it into a GeoTable.
2. Spatial Join (spatial_join) :
This method performs a spatial join between two GeoTable instances using geopandas.sjoin.
It simplifies the process by allowing users to specify the type of join (how) and the spatial predicate (predicate).
3. Conversion to GeoDataFrame (to_geodataframe) :
Ensures compatibility with geopandas for advanced geospatial operations.

- Example Usage
1. Input CSV File (`locations.csv`):
```
name,longitude,latitude
LocationA,-122.4194,37.7749
LocationB,-118.2437,34.0522
LocationC,-73.9352,40.7306
```

2. Input GeoJSON File (`regions.geojson`):

```
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": { "region": "West" },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[-125, 30], [-125, 40], [-115, 40], [-115, 30], [-125, 30]]]
      }
    },
    {
      "type": "Feature",
      "properties": { "region": "East" },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[-80, 35], [-80, 45], [-70, 45], [-70, 35], [-80, 35]]]
      }
    }
  ]
}
```

- Output
1. Joined Table :
```
name      | longitude | latitude | geometry                  | region
LocationA | -122.4194 | 37.7749  | POINT (-122.4194 37.7749) | West
LocationC | -73.9352  | 40.7306  | POINT (-73.9352 40.7306)  | East
```
2. Plot :
A map showing points colored by their associated regions.


# summary
- Explanation of the Workflow
1. Reading Data :
* The from_csv method reads the CSV file and converts longitude and latitude into geometry objects.
* The read_geojson method reads the GeoJSON file containing polygons.
2. Spatial Join :
* The spatial_join method uses geopandas.sjoin to join points with polygons based on spatial relationships like "within" or "intersects".
* The result is a new GeoTable that combines attributes from both datasets.
3. Visualization :
* The plot method visualizes the joined data, optionally coloring points by their associated regions.
- This implementation makes it easy for users to work with geospatial data without needing to understand the underlying complexities of geopandas.
